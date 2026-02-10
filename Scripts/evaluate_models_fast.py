"""
Fast evaluation of fusion models with single-pass feature caching.

Fixes over evaluate_models_ii.py:
- Encoder loaded once, features cached for all checkpoints
- Prior loaded once, prior_probs cached
- R-AUC uses vectorized numpy (not O(n^2) Python loop)
- Prior-only metrics computed from cache (no encoder needed)
- num_classes derived from data, not hardcoded

Architecture:
  Phase 1: Load FrozenAudioEncoder once -> extract features for entire val set
  Phase 2: Load PriorWrapper once -> get prior_probs for entire val set
  Phase 3: Compute R-AUC + NMI once (same frozen features for all models)
  Phase 4: Prior-only metrics from cached prior_probs (no encoder)
  Phase 5: Per-checkpoint: load small params -> apply classifier+fusion on cached features
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    normalized_mutual_info_score,
    average_precision_score,
)
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "NatureLM-audio"))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from Models import FrozenAudioEncoder, PriorGatingNetwork
from train_wf_stage_d import BirdAudioDataset, collate_fn, PriorWrapper


# ============================================================================
# Metrics
# ============================================================================


def compute_accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    """Top-1 accuracy via argmax."""
    return float((logits.argmax(axis=1) == labels).mean())


def compute_cmap(logits: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    """Class-mean Average Precision (macro-averaged, equal weight per class)."""
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    labels_onehot = np.zeros((len(labels), num_classes), dtype=np.float32)
    labels_onehot[np.arange(len(labels)), labels] = 1.0
    aps = []
    for c in range(num_classes):
        if labels_onehot[:, c].sum() == 0:
            continue
        try:
            aps.append(average_precision_score(labels_onehot[:, c], probs[:, c]))
        except ValueError:
            continue
    return float(np.mean(aps)) if aps else 0.0


def compute_retrieval_auc(features: np.ndarray, labels: np.ndarray) -> float:
    """Retrieval ROC-AUC via vectorized cosine similarity (no Python loop)."""
    features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    similarities = features_norm @ features_norm.T
    n = len(labels)
    y_true = labels[:, None] == labels[None, :]
    mask = ~np.eye(n, dtype=bool)
    try:
        return float(roc_auc_score(y_true[mask].ravel(), similarities[mask].ravel()))
    except ValueError:
        return 0.5


def compute_nmi(features: np.ndarray, labels: np.ndarray) -> float:
    """Normalized Mutual Information via k-means clustering."""
    n_clusters = len(np.unique(labels))
    features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    kmeans = KMeans(
        n_clusters=min(n_clusters, len(features)), random_state=42, n_init=10
    )
    cluster_labels = kmeans.fit_predict(features_norm)
    return float(normalized_mutual_info_score(labels, cluster_labels))


# ============================================================================
# Core Functions
# ============================================================================


@torch.no_grad()
def extract_features(encoder, dataloader, device):
    """Single pass through val set — cache encoder features."""
    encoder.eval()
    all_features, all_labels, all_metadata, all_indices = [], [], [], []
    for batch in tqdm(dataloader, desc="Extracting features"):
        features = encoder(batch["audio"].to(device))
        all_features.append(features.float().cpu())
        all_labels.append(batch["label"])
        all_metadata.extend(batch["metadata"])
        all_indices.extend(batch["sample_indices"])
    return {
        "features": torch.cat(all_features, dim=0),
        "labels": torch.cat(all_labels, dim=0).numpy(),
        "metadata": all_metadata,
        "sample_indices": all_indices,
    }


@torch.no_grad()
def evaluate_checkpoint(ckpt_path, encoder, cached, prior_probs_np, num_classes, device):
    """Evaluate one checkpoint on cached features (no audio re-encoding).

    Handles both old (WeightedFusionModel: audio_encoder/audio_classifier/w_weight)
    and new (StageBModel: encoder/classifier/gate_network) checkpoint formats.
    Only the small params (classifier + fusion) are loaded — encoder is skipped.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]

    # Detect format by key prefixes
    has_old_keys = any(k.startswith("audio_encoder.") for k in sd)
    has_gate = any(k.startswith("gate_network.") for k in sd)

    # --- Load classifier ---
    if has_old_keys:
        cls_w, cls_b = sd["audio_classifier.weight"], sd["audio_classifier.bias"]
    else:
        cls_w, cls_b = sd["classifier.weight"], sd["classifier.bias"]

    classifier = nn.Linear(cls_w.shape[1], cls_w.shape[0], device=device)
    classifier.weight.data.copy_(cls_w)
    classifier.bias.data.copy_(cls_b)

    # --- Prepare tensors ---
    features = cached["features"].to(device)
    labels = cached["labels"]
    metadata = cached["metadata"]
    prior_probs = torch.from_numpy(prior_probs_np).float().to(device)

    audio_logits = classifier(features.float())

    # --- Temperature scaling ---
    temp_val = sd.get("temperature", torch.tensor(1.0)).item()
    temp = max(abs(temp_val), 1e-8)
    scaled = audio_logits / temp

    # --- Prior log-probs ---
    eps_val = sd.get("epsilon", torch.tensor(0.0)).item()
    safe_eps = max(eps_val, -prior_probs.min().item() + 1e-8)
    log_prior = torch.log(prior_probs + safe_eps)

    # --- Fusion ---
    if has_gate:
        # Gating network: w(a, x, t)
        gate_sd = {k.removeprefix("gate_network."): v
                   for k, v in sd.items() if k.startswith("gate_network.")}
        gate = PriorGatingNetwork()
        gate.load_state_dict(gate_sd)
        gate.to(device).eval()

        # Compute gating features (same logic as StageBModel)
        probs = F.softmax(scaled, dim=1)
        a_max = probs.max(dim=1)[0]
        a_ent = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
        a_top2 = torch.topk(probs, k=2, dim=1)[0]
        af = torch.stack([a_max, a_ent, a_top2[:, 0] - a_top2[:, 1]], dim=1)

        p_max = prior_probs.max(dim=1)[0]
        p_ent = -(prior_probs * torch.log(prior_probs + 1e-10)).sum(dim=1)
        p_top2 = torch.topk(prior_probs, k=2, dim=1)[0]
        pf = torch.stack([p_max, p_ent, p_top2[:, 0] - p_top2[:, 1]], dim=1)

        w = gate(af, pf, metadata)
        final_logits = scaled + w.unsqueeze(1) * log_prior
        w_np = w.float().cpu().numpy()
    else:
        # Scalar weight
        w_val = sd.get("w_weight", torch.tensor(0.0)).item()
        final_logits = scaled + w_val * log_prior
        w_np = np.full(len(labels), w_val)

    final_np = final_logits.float().cpu().numpy()
    audio_np = audio_logits.float().cpu().numpy()

    return {
        "likelihood_acc": compute_accuracy(audio_np, labels),
        "likelihood_cmap": compute_cmap(audio_np, labels, num_classes),
        "posterior_acc": compute_accuracy(final_np, labels),
        "posterior_cmap": compute_cmap(final_np, labels, num_classes),
        "temperature": temp_val,
        "epsilon": eps_val,
        "w_mean": float(w_np.mean()),
        "w_std": float(w_np.std()),
        "w_min": float(w_np.min()),
        "w_max": float(w_np.max()),
    }


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Fast evaluation of fusion models")
    parser.add_argument("--data_dir", type=str, default="../Data/cbi")
    parser.add_argument("--priors_cache", type=str, required=True)
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "max", "cls"])
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- Data setup ---
    data_dir = Path(args.data_dir).resolve()
    train_csv = pd.read_csv(data_dir / "train.csv")

    audio_dir = data_dir / "train_audio"
    valid = []
    for _, row in tqdm(train_csv.iterrows(), total=len(train_csv), desc="Validating files"):
        p = audio_dir / row["ebird_code"] / row["filename"]
        valid.append(Path(str(p).replace(".mp3", ".ogg")).is_file())
    train_csv = train_csv[valid]
    print(f"Valid samples: {len(train_csv)}")

    all_species = sorted(train_csv["species"].unique())
    label_to_idx = {s: i for i, s in enumerate(all_species)}
    idx_to_label = {i: s for s, i in label_to_idx.items()}
    num_classes = len(all_species)
    print(f"Classes: {num_classes}")

    _, val_df = train_test_split(
        train_csv,
        test_size=args.val_split,
        stratify=train_csv["species"],
        random_state=args.seed,
    )
    val_dataset = BirdAudioDataset(audio_dir, val_df, label_to_idx)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # --- Phase 1: Extract features (encoder loaded once) ---
    print("\n[Phase 1] Loading encoder and extracting features...")
    encoder = FrozenAudioEncoder(pooling=args.pooling).to(device)
    cached = extract_features(encoder, val_loader, device)
    print(f"Cached {len(cached['labels'])} samples, feature dim {cached['features'].shape[1]}")

    # --- Phase 2: Get prior probs (loaded once) ---
    print("\n[Phase 2] Loading priors...")
    prior = PriorWrapper(
        species_list=list(idx_to_label.values()),
        cache_path=Path(args.priors_cache).resolve(),
    )
    prior_probs_np = prior.get_batch(cached["sample_indices"])
    print(f"Prior shape: {prior_probs_np.shape}")

    # --- Phase 3: Encoder-level metrics (computed once, shared across models) ---
    print("\n[Phase 3] Computing R-AUC and NMI (encoder features, computed once)...")
    features_np = cached["features"].numpy()
    labels_np = cached["labels"]
    rauc = compute_retrieval_auc(features_np, labels_np)
    nmi = compute_nmi(features_np, labels_np)
    print(f"R-AUC: {rauc:.4f}, NMI: {nmi:.4f}")

    # --- Phase 4: Prior-only metrics (no encoder needed) ---
    print("\n[Phase 4] Prior-only metrics...")
    log_prior_np = np.log(prior_probs_np + 1e-10)
    prior_acc = compute_accuracy(log_prior_np, labels_np)
    prior_cmap = compute_cmap(log_prior_np, labels_np, num_classes)
    print(f"Prior Acc: {100 * prior_acc:.2f}%, Prior cmAP: {prior_cmap:.4f}")

    # --- Phase 5: Per-checkpoint evaluation (cached features, no re-encoding) ---
    print("\n[Phase 5] Evaluating checkpoints...")
    ckpt_results = {}
    for ckpt_path in args.checkpoints:
        name = Path(ckpt_path).stem
        print(f"\n  Evaluating: {name}")
        m = evaluate_checkpoint(
            ckpt_path, encoder, cached, prior_probs_np, num_classes, device
        )
        ckpt_results[name] = m
        print(
            f"    Likelihood  Acc={100 * m['likelihood_acc']:.2f}%  "
            f"cmAP={m['likelihood_cmap']:.4f}"
        )
        print(
            f"    Posterior   Acc={100 * m['posterior_acc']:.2f}%  "
            f"cmAP={m['posterior_cmap']:.4f}"
        )
        print(
            f"    w: mean={m['w_mean']:.4f} std={m['w_std']:.4f} "
            f"min={m['w_min']:.4f} max={m['w_max']:.4f}"
        )
        print(f"    T={m['temperature']:.4f}  eps={m['epsilon']:.6f}")

        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Summary table ---
    print(f"\n{'=' * 90}")
    print(
        f"{'Model':<35} {'Acc':>8} {'cmAP':>8} {'R-AUC':>8} {'NMI':>8}"
    )
    print(f"{'-' * 90}")
    print(
        f"{'Prior only':<35} {100 * prior_acc:>7.2f}% {prior_cmap:>8.4f}"
        f" {'N/A':>8} {'N/A':>8}"
    )
    for name, m in ckpt_results.items():
        print(
            f"{name + ' (likelihood)':<35} "
            f"{100 * m['likelihood_acc']:>7.2f}% {m['likelihood_cmap']:>8.4f}"
            f" {rauc:>8.4f} {nmi:>8.4f}"
        )
        print(
            f"{name + ' (posterior)':<35} "
            f"{100 * m['posterior_acc']:>7.2f}% {m['posterior_cmap']:>8.4f}"
            f" {rauc:>8.4f} {nmi:>8.4f}"
        )
    print(f"{'=' * 90}")

    # --- Save results ---
    all_results = {
        "rauc": rauc,
        "nmi": nmi,
        "prior_acc": prior_acc,
        "prior_cmap": prior_cmap,
        "checkpoints": ckpt_results,
        "num_classes": num_classes,
        "num_val_samples": len(labels_np),
    }
    torch.save(all_results, "eval_results.pth")
    print("Results saved to eval_results.pth")


if __name__ == "__main__":
    main()
