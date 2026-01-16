"""
Evaluate and compare models: NatureLM likelihood, Prior, Posterior Stage A, Posterior Stage B.

Creates:
1. Comparison table of Probe, R-AUC, NMI
2. Confusion matrices (TP, FP, FN, TN) for each model
3. Examples where likelihood is wrong but posterior is correct (saved by prior)
"""

import argparse
import warnings
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_auc_score, 
    normalized_mutual_info_score
)
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

# Add NatureLM to path
sys.path.insert(0, str(Path(__file__).parent.parent / "NatureLM-audio"))

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
    try:
        import seaborn as sns
        HAS_SEABORN = True
    except ImportError:
        HAS_SEABORN = False
        print("Warning: seaborn not installed, some plots may look less polished")
except ImportError as e:
    HAS_MATPLOTLIB = False
    HAS_SEABORN = False
    print(f"Warning: matplotlib not installed ({e}), visualizations disabled")

# Import from train_weighted_fusion
from train_weighted_fusion import (
    CBIBirdDataset, collate_fn, NatureLMAudioEncoder, 
    WeightedFusionModel, EBirdPriorWrapper
)


def compute_probe_accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    """Probe accuracy = top-1 accuracy from logits."""
    preds = logits.argmax(axis=1)
    return accuracy_score(labels, preds)


def compute_retrieval_auc(features: np.ndarray, labels: np.ndarray) -> float:
    """Compute retrieval ROC-AUC using cosine similarity."""
    features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    similarities = features_norm @ features_norm.T
    n = len(labels)
    y_true = []
    y_score = []
    for i in range(n):
        for j in range(n):
            if i != j:
                y_true.append(1 if labels[i] == labels[j] else 0)
                y_score.append(similarities[i, j])
    try:
        return roc_auc_score(y_true, y_score)
    except:
        return 0.5


def compute_nmi(features: np.ndarray, labels: np.ndarray) -> float:
    """Compute Normalized Mutual Information using k-means clustering."""
    n_clusters = len(np.unique(labels))
    features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    kmeans = KMeans(n_clusters=min(n_clusters, len(features)), random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_norm)
    from sklearn.metrics import normalized_mutual_info_score
    return normalized_mutual_info_score(labels, cluster_labels)


def compute_confusion_metrics(y_true, y_pred, num_classes):
    """Compute TP, FP, FN, TN from confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)
    return {
        "tp": tp.sum(),
        "fp": fp.sum(),
        "fn": fn.sum(),
        "tn": tn.sum(),
        "fpr": fp.sum() / (fp.sum() + tn.sum()) if (fp.sum() + tn.sum()) > 0 else 0.0,
        "fnr": fn.sum() / (tp.sum() + fn.sum()) if (tp.sum() + fn.sum()) > 0 else 0.0,
    }


@torch.no_grad()
def evaluate_model(model, dataloader, device, prior_model, idx_to_label):
    """Evaluate a single model and return metrics and predictions."""
    model.eval()
    
    all_final_logits = []
    all_audio_logits = []
    all_prior_probs = []
    all_features = []
    all_labels = []
    all_metadata = []
    all_weights = []  # For gating weights
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        audio = batch["audio"].to(device)
        labels = batch["label"]
        metadata = batch["metadata"]
        
        # Get prior
        latitudes = [m["latitude"] for m in metadata]
        longitudes = [m["longitude"] for m in metadata]
        dates = [m["date"] for m in metadata]
        sample_indices = batch.get("sample_indices", None)
        prior_probs = prior_model.get_prior_probs_batch(
            latitudes, longitudes, dates, sample_indices=sample_indices
        )
        prior_probs_tensor = torch.from_numpy(prior_probs).float().to(device)
        
        with autocast("cuda", dtype=torch.bfloat16):
            final_logits, audio_logits, prior_robust = model(
                audio, prior_probs_tensor, metadata if model.use_gating else None
            )
            features = model.audio_encoder(audio)
            
            # Get gating weights if available
            if model.use_gating and hasattr(model, 'gate_network'):
                audio_features = model.compute_audio_features(audio_logits)
                prior_features = model.compute_prior_features(prior_probs_tensor)
                weights = model.gate_network(audio_features, prior_features, metadata)
                all_weights.append(weights.float().cpu().detach())
            elif hasattr(model, 'w_weight'):
                batch_size = audio.size(0)
                weights = model.w_weight.expand(batch_size)
                all_weights.append(weights.cpu().detach())
        
        all_final_logits.append(final_logits.float().cpu().detach())
        all_audio_logits.append(audio_logits.float().cpu().detach())
        all_prior_probs.append(prior_robust.float().cpu().detach())
        all_features.append(features.float().cpu().detach())
        all_labels.append(labels.detach() if isinstance(labels, torch.Tensor) else labels)
        all_metadata.extend(metadata)
    
    final_logits = torch.cat(all_final_logits, dim=0).numpy()
    audio_logits = torch.cat(all_audio_logits, dim=0).numpy()
    prior_probs = torch.cat(all_prior_probs, dim=0).numpy()
    features = torch.cat(all_features, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0)
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().numpy()
    else:
        labels = np.array(labels)
    
    weights = torch.cat(all_weights, dim=0).numpy() if all_weights else None
    
    # Compute predictions
    final_preds = final_logits.argmax(axis=1)
    audio_preds = audio_logits.argmax(axis=1)
    prior_preds = prior_probs.argmax(axis=1)
    
    num_classes = len(np.unique(labels))
    
    # Compute metrics
    metrics = {
        "likelihood_probe": compute_probe_accuracy(audio_logits, labels),
        "likelihood_rauc": compute_retrieval_auc(features, labels),
        "likelihood_nmi": compute_nmi(features, labels),
        "posterior_probe": compute_probe_accuracy(final_logits, labels),
        "posterior_rauc": compute_retrieval_auc(features, labels),
        "posterior_nmi": compute_nmi(features, labels),
        "prior_probe": compute_probe_accuracy(np.log(prior_probs + 1e-10), labels),
    }
    
    # Confusion matrices
    likelihood_cm = compute_confusion_metrics(labels, audio_preds, num_classes)
    posterior_cm = compute_confusion_metrics(labels, final_preds, num_classes)
    prior_cm = compute_confusion_metrics(labels, prior_preds, num_classes)
    
    metrics.update({
        "likelihood_tp": likelihood_cm["tp"],
        "likelihood_fp": likelihood_cm["fp"],
        "likelihood_fn": likelihood_cm["fn"],
        "likelihood_tn": likelihood_cm["tn"],
        "posterior_tp": posterior_cm["tp"],
        "posterior_fp": posterior_cm["fp"],
        "posterior_fn": posterior_cm["fn"],
        "posterior_tn": posterior_cm["tn"],
        "prior_tp": prior_cm["tp"],
        "prior_fp": prior_cm["fp"],
        "prior_fn": prior_cm["fn"],
        "prior_tn": prior_cm["tn"],
    })
    
    return {
        "metrics": metrics,
        "audio_logits": audio_logits,
        "prior_probs": prior_probs,
        "final_logits": final_logits,
        "labels": labels,
        "weights": weights,
        "metadata": all_metadata,
    }


def plot_comparison_table(metrics_dict, output_path):
    """Create comparison table of Probe, R-AUC, NMI."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping table plot")
        return
    
    models = []
    probe_values = []
    rauc_values = []
    nmi_values = []
    
    # NatureLM (likelihood)
    if "likelihood" in metrics_dict:
        models.append("NatureLM\n(Likelihood)")
        probe_values.append(metrics_dict["likelihood"]["metrics"]["likelihood_probe"])
        rauc_values.append(metrics_dict["likelihood"]["metrics"]["likelihood_rauc"])
        nmi_values.append(metrics_dict["likelihood"]["metrics"]["likelihood_nmi"])
    
    # Prior
    if "prior" in metrics_dict:
        models.append("Prior Model")
        probe_values.append(metrics_dict["prior"]["metrics"]["prior_probe"])
        rauc_values.append(np.nan)  # R-AUC not applicable to prior
        nmi_values.append(np.nan)  # NMI not applicable to prior
    
    # Posterior Stage A
    if "stage_a" in metrics_dict:
        models.append("Posterior\nStage A")
        probe_values.append(metrics_dict["stage_a"]["metrics"]["posterior_probe"])
        rauc_values.append(metrics_dict["stage_a"]["metrics"]["posterior_rauc"])
        nmi_values.append(metrics_dict["stage_a"]["metrics"]["posterior_nmi"])
    
    # Posterior Stage B
    if "stage_b" in metrics_dict:
        models.append("Posterior\nStage B")
        probe_values.append(metrics_dict["stage_b"]["metrics"]["posterior_probe"])
        rauc_values.append(metrics_dict["stage_b"]["metrics"]["posterior_rauc"])
        nmi_values.append(metrics_dict["stage_b"]["metrics"]["posterior_nmi"])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    data = []
    for i, model in enumerate(models):
        probe_str = f"{100*probe_values[i]:.2f}%" if not np.isnan(probe_values[i]) else "N/A"
        rauc_str = f"{rauc_values[i]:.4f}" if not np.isnan(rauc_values[i]) else "N/A"
        nmi_str = f"{nmi_values[i]:.4f}" if not np.isnan(nmi_values[i]) else "N/A"
        data.append([model, probe_str, rauc_str, nmi_str])
    
    table = ax.table(cellText=data,
                     colLabels=["Model", "Probe", "R-AUC", "NMI"],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title("Model Comparison: Probe, R-AUC, NMI", fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison table to {output_path}")


def plot_confusion_matrices(metrics_dict, output_path):
    """Create confusion matrices showing TP, FP, FN, TN."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping confusion matrices")
        return
    
    n_models = sum(1 for k in ["likelihood", "prior", "stage_a", "stage_b"] if k in metrics_dict)
    if n_models == 0:
        return
    
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
    if n_models == 1:
        axes = [axes]
    
    idx = 0
    
    # NatureLM
    if "likelihood" in metrics_dict:
        m = metrics_dict["likelihood"]["metrics"]
        data = [[m["likelihood_tp"], m["likelihood_fp"]],
                [m["likelihood_fn"], m["likelihood_tn"]]]
        if HAS_SEABORN:
            sns.heatmap(data, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                        xticklabels=['Pred: Positive', 'Pred: Negative'],
                        yticklabels=['True: Positive', 'True: Negative'],
                        cbar=False)
        else:
            im = axes[idx].imshow(data, cmap='Blues', aspect='auto')
            axes[idx].set_xticks([0, 1])
            axes[idx].set_yticks([0, 1])
            axes[idx].set_xticklabels(['Pred: Positive', 'Pred: Negative'])
            axes[idx].set_yticklabels(['True: Positive', 'True: Negative'])
            for i in range(2):
                for j in range(2):
                    axes[idx].text(j, i, str(data[i][j]), ha='center', va='center', color='white' if data[i][j] > data[i].mean() else 'black')
            plt.colorbar(im, ax=axes[idx])
        axes[idx].set_title("NatureLM\n(Likelihood)", fontweight='bold')
        idx += 1
    
    # Prior
    if "prior" in metrics_dict:
        m = metrics_dict["prior"]["metrics"]
        data = [[m["prior_tp"], m["prior_fp"]],
                [m["prior_fn"], m["prior_tn"]]]
        if HAS_SEABORN:
            sns.heatmap(data, annot=True, fmt='d', cmap='Greens', ax=axes[idx],
                        xticklabels=['Pred: Positive', 'Pred: Negative'],
                        yticklabels=['True: Positive', 'True: Negative'],
                        cbar=False)
        else:
            im = axes[idx].imshow(data, cmap='Greens', aspect='auto')
            axes[idx].set_xticks([0, 1])
            axes[idx].set_yticks([0, 1])
            axes[idx].set_xticklabels(['Pred: Positive', 'Pred: Negative'])
            axes[idx].set_yticklabels(['True: Positive', 'True: Negative'])
            for i in range(2):
                for j in range(2):
                    axes[idx].text(j, i, str(data[i][j]), ha='center', va='center', color='white' if data[i][j] > data[i].mean() else 'black')
            plt.colorbar(im, ax=axes[idx])
        axes[idx].set_title("Prior Model", fontweight='bold')
        idx += 1
    
    # Stage A
    if "stage_a" in metrics_dict:
        m = metrics_dict["stage_a"]["metrics"]
        data = [[m["posterior_tp"], m["posterior_fp"]],
                [m["posterior_fn"], m["posterior_tn"]]]
        if HAS_SEABORN:
            sns.heatmap(data, annot=True, fmt='d', cmap='Oranges', ax=axes[idx],
                        xticklabels=['Pred: Positive', 'Pred: Negative'],
                        yticklabels=['True: Positive', 'True: Negative'],
                        cbar=False)
        else:
            im = axes[idx].imshow(data, cmap='Oranges', aspect='auto')
            axes[idx].set_xticks([0, 1])
            axes[idx].set_yticks([0, 1])
            axes[idx].set_xticklabels(['Pred: Positive', 'Pred: Negative'])
            axes[idx].set_yticklabels(['True: Positive', 'True: Negative'])
            for i in range(2):
                for j in range(2):
                    axes[idx].text(j, i, str(data[i][j]), ha='center', va='center', color='white' if data[i][j] > data[i].mean() else 'black')
            plt.colorbar(im, ax=axes[idx])
        axes[idx].set_title("Posterior\nStage A", fontweight='bold')
        idx += 1
    
    # Stage B
    if "stage_b" in metrics_dict:
        m = metrics_dict["stage_b"]["metrics"]
        data = [[m["posterior_tp"], m["posterior_fp"]],
                [m["posterior_fn"], m["posterior_tn"]]]
        if HAS_SEABORN:
            sns.heatmap(data, annot=True, fmt='d', cmap='Purples', ax=axes[idx],
                        xticklabels=['Pred: Positive', 'Pred: Negative'],
                        yticklabels=['True: Positive', 'True: Negative'],
                        cbar=False)
        else:
            im = axes[idx].imshow(data, cmap='Purples', aspect='auto')
            axes[idx].set_xticks([0, 1])
            axes[idx].set_yticks([0, 1])
            axes[idx].set_xticklabels(['Pred: Positive', 'Pred: Negative'])
            axes[idx].set_yticklabels(['True: Positive', 'True: Negative'])
            for i in range(2):
                for j in range(2):
                    axes[idx].text(j, i, str(data[i][j]), ha='center', va='center', color='white' if data[i][j] > data[i].mean() else 'black')
            plt.colorbar(im, ax=axes[idx])
        axes[idx].set_title("Posterior\nStage B", fontweight='bold')
        idx += 1
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrices to {output_path}")


def plot_pathological_examples(
    audio_logits, prior_probs, final_logits, labels, weights, metadata,
    idx_to_label, output_dir, num_examples=5
):
    """Plot examples where likelihood is wrong but posterior is correct."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping pathological examples")
        return
    
    # Convert to probabilities
    audio_probs = torch.softmax(torch.from_numpy(audio_logits), dim=1).numpy()
    final_probs = torch.softmax(torch.from_numpy(final_logits), dim=1).numpy()
    
    # Find examples where:
    # - Likelihood prediction is wrong
    # - Posterior prediction is correct
    audio_preds = audio_probs.argmax(axis=1)
    final_preds = final_probs.argmax(axis=1)
    
    mask = (audio_preds != labels) & (final_preds == labels)
    candidate_indices = np.where(mask)[0]
    
    if len(candidate_indices) == 0:
        print("No examples found where likelihood is wrong but posterior is correct")
        return
    
    # Sort by weight (highest weight = most prior influence)
    weights_flat = None
    if weights is not None:
        weights_flat = weights.flatten() if len(weights.shape) > 1 else weights
        candidate_weights = weights_flat[candidate_indices]
        sorted_idx = np.argsort(candidate_weights)[::-1]  # Highest first
        candidate_indices = candidate_indices[sorted_idx]
    
    num_examples = min(num_examples, len(candidate_indices))
    
    for i in range(num_examples):
        idx = candidate_indices[i]
        
        true_label = labels[idx]
        audio_pred = audio_preds[idx]
        final_pred = final_preds[idx]
        
        # Get top species
        top_k = 10
        top_indices = np.argsort(final_probs[idx])[-top_k:][::-1]
        
        species_names = [idx_to_label[j] for j in top_indices]
        audio_vals = audio_probs[idx, top_indices]
        prior_vals = prior_probs[idx, top_indices]
        final_vals = final_probs[idx, top_indices]
        
        # Normalize prior for display
        prior_vals_norm = prior_vals / (prior_vals.sum() + 1e-10)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(species_names))
        width = 0.25
        
        bars1 = ax.bar(x - width, audio_vals, width, label='Likelihood (NatureLM)', 
                      color='#3498db', alpha=0.8)
        bars2 = ax.bar(x, prior_vals_norm, width, label='Prior', 
                      color='#2ecc71', alpha=0.8)
        bars3 = ax.bar(x + width, final_vals, width, label='Posterior', 
                      color='#e74c3c', alpha=0.8)
        
        # Highlight true label and predictions
        true_idx_in_top = np.where(top_indices == true_label)[0]
        audio_pred_idx_in_top = np.where(top_indices == audio_pred)[0]
        final_pred_idx_in_top = np.where(top_indices == final_pred)[0]
        
        if len(true_idx_in_top) > 0:
            ax.axvline(x=true_idx_in_top[0], color='green', linestyle='--', linewidth=2, alpha=0.5, label='True label')
        if len(audio_pred_idx_in_top) > 0:
            ax.axvline(x=audio_pred_idx_in_top[0], color='blue', linestyle=':', linewidth=2, alpha=0.5, label='Likelihood pred')
        if len(final_pred_idx_in_top) > 0:
            ax.axvline(x=final_pred_idx_in_top[0], color='red', linestyle='-.', linewidth=2, alpha=0.5, label='Posterior pred')
        
        ax.set_xlabel('Species', fontsize=12, fontweight='bold')
        ax.set_ylabel('Probability', fontsize=12, fontweight='bold')
        ax.set_title(f'Example {i+1}: Prior "Saves" Likelihood\n'
                    f'True: {idx_to_label[true_label]} | '
                    f'Likelihood: {idx_to_label[audio_pred]} (WRONG) | '
                    f'Posterior: {idx_to_label[final_pred]} (CORRECT)',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(species_names, rotation=45, ha='right', fontsize=10)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        # Add metadata
        meta = metadata[idx]
        date_str = meta.get("date", "Unknown")
        lat = meta.get("latitude", "?")
        lon = meta.get("longitude", "?")
        weight_str = f"{weights_flat[idx]:.3f}" if weights_flat is not None else "N/A"
        
        info_text = f"Date: {date_str}\nLocation: ({lat:.2f}, {lon:.2f})\nWeight: {weight_str}"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        output_path = output_dir / f"pathological_example_{i+1}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved pathological example {i+1} to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare models")
    parser.add_argument("--data_dir", type=str, default="../Data/cbi")
    parser.add_argument("--priors_dir", type=str, default="../Data/priors")
    parser.add_argument("--priors_cache", type=str, default=None)
    parser.add_argument("--checkpoint_stage_a", type=str, default=None,
                       help="Path to Stage A checkpoint (scalar weight). If not provided, will try common locations.")
    parser.add_argument("--checkpoint_stage_b", type=str, default="../checkpoints_fusion/best_model.pth",
                       help="Path to Stage B checkpoint (gating network)")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_examples", type=int, default=5)
    parser.add_argument("--use_qformer", action="store_true")
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "max", "cls"])
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data_dir = Path(args.data_dir).resolve()
    train_csv = pd.read_csv(data_dir / "train.csv")
    
    all_species = sorted(train_csv["species"].unique())
    label_to_idx = {s: i for i, s in enumerate(all_species)}
    idx_to_label = {i: s for s, i in label_to_idx.items()}
    num_classes = len(all_species)
    
    # Split data (use validation set)
    train_split, val_split = train_test_split(
        train_csv, test_size=0.1, stratify=train_csv["species"], random_state=args.seed
    )
    
    audio_dir = data_dir / "train_audio"
    val_dataset = CBIBirdDataset(audio_dir, val_split, label_to_idx)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn
    )
    
    # Load prior model
    print("Loading prior model...")
    cache_path = Path(args.priors_cache).resolve() if args.priors_cache else None
    prior_model = EBirdPriorWrapper(
        priors_dir=Path(args.priors_dir).resolve(),
        species_list=list(idx_to_label.values()),
        cbi_metadata_path=data_dir / "train.csv",
        cache_path=cache_path,
        df_index=train_csv.index,
    )
    
    # Load models
    metrics_dict = {}
    
    # We'll extract likelihood metrics from Stage B model (both use same audio encoder)
    # So we don't need a separate likelihood model
    
    # Stage A (scalar weight)
    stage_a_path = args.checkpoint_stage_a
    if stage_a_path is None:
        # Try common locations
        possible_paths = [
            "../checkpoints_fusion_stage_a/best_model.pth",
            "../checkpoints_fusion/best_model_stage_a.pth",
        ]
        for p in possible_paths:
            if Path(p).exists():
                stage_a_path = p
                break
    
    if stage_a_path and Path(stage_a_path).exists():
        print("\n" + "="*60)
        print("Evaluating Posterior Stage A")
        print("="*60)
        # Clear GPU cache before loading
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            import time
            time.sleep(1)  # Small delay to let CUDA settle
        
        audio_encoder_a = NatureLMAudioEncoder(use_qformer=args.use_qformer, pooling=args.pooling)
        model_a = WeightedFusionModel(
            audio_encoder=audio_encoder_a,
            num_classes=num_classes,
            use_gating=False,
        )
        model_a = model_a.to(device)
        checkpoint = torch.load(stage_a_path, map_location=device)
        model_a.load_state_dict(checkpoint["model_state_dict"])
        results_a = evaluate_model(model_a, val_loader, device, prior_model, idx_to_label)
        metrics_dict["stage_a"] = results_a
        
        # Clean up Stage A model
        del model_a
        del audio_encoder_a
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            import time
            time.sleep(1)
        
        # If likelihood not already extracted from Stage B, extract from Stage A
        if "likelihood" not in metrics_dict:
            metrics_dict["likelihood"] = {
                "metrics": {
                    "likelihood_probe": results_a["metrics"]["likelihood_probe"],
                    "likelihood_rauc": results_a["metrics"]["likelihood_rauc"],
                    "likelihood_nmi": results_a["metrics"]["likelihood_nmi"],
                    "likelihood_tp": results_a["metrics"]["likelihood_tp"],
                    "likelihood_fp": results_a["metrics"]["likelihood_fp"],
                    "likelihood_fn": results_a["metrics"]["likelihood_fn"],
                    "likelihood_tn": results_a["metrics"]["likelihood_tn"],
                }
            }
    else:
        print(f"Warning: Stage A checkpoint not found")
    
    # Stage B (gating network)
    if Path(args.checkpoint_stage_b).exists():
        print("\n" + "="*60)
        print("Evaluating Posterior Stage B")
        print("="*60)
        # Clear GPU cache before loading
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            import time
            time.sleep(1)  # Small delay to let CUDA settle
        
        audio_encoder_b = NatureLMAudioEncoder(use_qformer=args.use_qformer, pooling=args.pooling)
        model_b = WeightedFusionModel(
            audio_encoder=audio_encoder_b,
            num_classes=num_classes,
            use_gating=True,
        )
        model_b = model_b.to(device)
        checkpoint = torch.load(args.checkpoint_stage_b, map_location=device)
        model_b.load_state_dict(checkpoint["model_state_dict"])
        results_b = evaluate_model(model_b, val_loader, device, prior_model, idx_to_label)
        metrics_dict["stage_b"] = results_b
        
        # Clean up Stage B model (but keep results for pathological examples)
        # We'll delete after we extract what we need
        del model_b
        del audio_encoder_b
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            import time
            time.sleep(1)
        
        # Extract likelihood metrics from Stage B (same audio encoder)
        metrics_dict["likelihood"] = {
            "metrics": {
                "likelihood_probe": results_b["metrics"]["likelihood_probe"],
                "likelihood_rauc": results_b["metrics"]["likelihood_rauc"],
                "likelihood_nmi": results_b["metrics"]["likelihood_nmi"],
                "likelihood_tp": results_b["metrics"]["likelihood_tp"],
                "likelihood_fp": results_b["metrics"]["likelihood_fp"],
                "likelihood_fn": results_b["metrics"]["likelihood_fn"],
                "likelihood_tn": results_b["metrics"]["likelihood_tn"],
            }
        }
    else:
        print(f"Warning: Stage B checkpoint not found at {args.checkpoint_stage_b}")
    
    # Prior model (evaluate separately)
    print("\n" + "="*60)
    print("Evaluating Prior Model")
    print("="*60)
    # Clear GPU cache before loading
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        import time
        time.sleep(1)  # Small delay to let CUDA settle
    
    # Create a dummy model that just returns prior
    audio_encoder_prior = NatureLMAudioEncoder(use_qformer=args.use_qformer, pooling=args.pooling)
    prior_only_model = WeightedFusionModel(
        audio_encoder=audio_encoder_prior,
        num_classes=num_classes,
        use_gating=False,
    )
    prior_only_model = prior_only_model.to(device)
    # Set weight to very high so prior dominates
    with torch.no_grad():
        prior_only_model.w_weight.fill_(10.0)
        prior_only_model.temperature.fill_(1.0)
    results_prior = evaluate_model(prior_only_model, val_loader, device, prior_model, idx_to_label)
    metrics_dict["prior"] = results_prior
    
    # Clean up Prior model
    del prior_only_model
    del audio_encoder_prior
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Create visualizations
    print("\n" + "="*60)
    print("Creating Visualizations")
    print("="*60)
    
    # Comparison table
    plot_comparison_table(metrics_dict, output_dir / "comparison_table.png")
    
    # Confusion matrices
    plot_confusion_matrices(metrics_dict, output_dir / "confusion_matrices.png")
    
    # Pathological examples (use Stage B if available, else Stage A)
    if "stage_b" in metrics_dict:
        results = metrics_dict["stage_b"]
        model_name = "Stage B"
    elif "stage_a" in metrics_dict:
        results = metrics_dict["stage_a"]
        model_name = "Stage A"
    else:
        print("No posterior model available for pathological examples")
        results = None
    
    if results is not None:
        plot_pathological_examples(
            results["audio_logits"],
            results["prior_probs"],
            results["final_logits"],
            results["labels"],
            results["weights"],
            results["metadata"],
            idx_to_label,
            output_dir,
            num_examples=args.num_examples
        )
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()