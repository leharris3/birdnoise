#!/usr/bin/env python3
"""
Gating weight (w) sweep experiment for bio-acoustic classification.

Trains 7 models on a 1000-sample subset of the CBI dataset:
- 6 models with fixed gating values: [0.0, 0.2, 0.4, 0.8, 1.6, 2.0]
- 1 model with adaptive gating network (learns w from spatio-temporal/ecological priors)

All models share a single trained linear probe (Phase 1).
Fixed-w models skip Phase 2 and evaluate directly with constant w.
Adaptive model trains the full gating network in Phase 2.

Usage:
    python w_sweep.py
    python w_sweep.py --subset_size 500 --seed 123
    python w_sweep.py --wandb_project my_project

    # Resume only adaptive gating (Phase 2) from a previous run:
    python w_sweep.py --adaptive_only --probe_checkpoint path/to/probe.pt
"""

import argparse
import json
import math
import os
import random
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm

# Add parent directories to path
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "NatureLM-audio"))
sys.path.insert(0, str(SCRIPT_DIR / "train"))

from datasets import CBIDataset, collate_fn

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Optional imports
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

try:
    from sklearn.metrics import (
        accuracy_score,
        top_k_accuracy_score,
        average_precision_score,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# Fixed gating values to sweep
FIXED_W_VALUES = [0.0, 0.2, 0.4, 0.8, 1.6, 2.0]


# ============================================================================
# Reproducibility
# ============================================================================


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# Models
# ============================================================================


class FrozenAudioEncoder(nn.Module):
    """Frozen NatureLM audio encoder."""

    def __init__(self, pooling: str = "mean"):
        super().__init__()
        self.pooling = pooling

        print("Loading NatureLM-audio encoder...")
        from NatureLM.models import NatureLM
        self.naturelm = NatureLM.from_pretrained("EarthSpeciesProject/NatureLM-audio")

        # Freeze all parameters
        for param in self.naturelm.parameters():
            param.requires_grad = False
        self.naturelm.eval()

        # Get encoder dimension
        self.encoder_dim = self.naturelm.llama_model.config.hidden_size
        print(f"Encoder dimension: {self.encoder_dim}")

    @torch.no_grad()
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract pooled features from audio."""
        self.naturelm.eval()
        device = next(self.naturelm.parameters()).device
        audio = audio.to(device)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            audio_embeds, audio_atts = self.naturelm.encode_audio(audio)

        if self.pooling == "mean":
            mask = audio_atts.unsqueeze(-1).float()
            features = (audio_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        elif self.pooling == "max":
            features = audio_embeds.max(dim=1)[0]
        else:  # cls
            features = audio_embeds[:, 0, :]

        return features


class PriorGatingNetwork(nn.Module):
    """
    Gating network to learn w(a,x,t) based on audio and prior features.

    Input: 13-dim vector
        - 3 audio features: max_prob, entropy, margin
        - 3 prior features: max_prob, entropy, margin
        - 7 metadata features: sin/cos(day), sin/cos(hour), lat, lon, dataset_type

    Output: scalar w in [0, w_max]
    """

    def __init__(
        self,
        input_dim: int = 13,
        hidden_dim: int = 64,
        w_max: float = 2.0,
        init_w: float = 0.0308,
    ):
        super().__init__()
        self.w_max = w_max
        self.init_w = init_w

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Initialize final layer bias for target initial w
        target_sigmoid = init_w / w_max
        target_sigmoid = max(1e-6, min(1 - 1e-6, target_sigmoid))
        init_bias = math.log(target_sigmoid / (1 - target_sigmoid))

        final_layer = self.mlp[-1]
        nn.init.zeros_(final_layer.weight)
        nn.init.constant_(final_layer.bias, init_bias)

    def forward(
        self,
        audio_features: torch.Tensor,
        prior_features: torch.Tensor,
        metadata: List[Dict],
        dataset_names: Optional[List[str]] = None,
    ) -> torch.Tensor:
        device = audio_features.device

        def safe_float_list(key, default=0.0):
            result = []
            for m in metadata:
                val = m.get(key, default)
                try:
                    result.append(float(val))
                except (ValueError, TypeError):
                    result.append(float(default))
            return result

        day_of_year = torch.tensor(
            safe_float_list("day_of_year", 182), dtype=torch.float32, device=device
        )
        hour = torch.tensor(
            safe_float_list("hour", 12), dtype=torch.float32, device=device
        )
        lat = torch.tensor(
            safe_float_list("latitude", 0.0), dtype=torch.float32, device=device
        )
        lon = torch.tensor(
            safe_float_list("longitude", 0.0), dtype=torch.float32, device=device
        )

        # Normalize
        lat_norm = lat / 90.0
        lon_norm = lon / 180.0

        # Cyclical encoding
        day_sin = torch.sin(2 * math.pi * day_of_year / 365.25)
        day_cos = torch.cos(2 * math.pi * day_of_year / 365.25)
        hour_sin = torch.sin(2 * math.pi * hour / 24.0)
        hour_cos = torch.cos(2 * math.pi * hour / 24.0)

        # Dataset type indicator (0 = BEANS/local time)
        ds_type = torch.zeros(len(metadata), dtype=torch.float32, device=device)

        meta_features = torch.stack(
            [day_sin, day_cos, hour_sin, hour_cos, lat_norm, lon_norm, ds_type], dim=1
        )
        combined = torch.cat([audio_features, prior_features, meta_features], dim=1)

        gate_logit = self.mlp(combined).squeeze(-1)
        w = torch.sigmoid(gate_logit) * self.w_max

        return w


class FusionModel(nn.Module):
    """
    Fusion model with optional gating network.

    final_logits = audio_logits / temperature + w * log(prior + epsilon)

    Supports both:
    - Fixed w mode: w is a constant value
    - Adaptive w mode: w is computed by the gating network
    """

    def __init__(
        self,
        encoder: FrozenAudioEncoder,
        num_classes: int,
        w_max: float = 100.0,
        init_w: float = 0.0308,
        init_temperature: float = 0.5101,
        init_epsilon: float = -0.049955,
        gate_hidden_dim: int = 64,
    ):
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes

        # Linear probe (classifier head)
        self.classifier = nn.Linear(encoder.encoder_dim, num_classes)

        # Learnable fusion parameters
        self.temperature = nn.Parameter(torch.tensor(init_temperature))
        self.epsilon = nn.Parameter(torch.tensor(init_epsilon))

        # Gating network (for adaptive mode)
        self.gate_network = PriorGatingNetwork(
            input_dim=13,
            hidden_dim=gate_hidden_dim,
            w_max=w_max,
            init_w=init_w,
        )

        # Fixed w mode
        self._fixed_w: Optional[float] = None

    def set_fixed_w(self, w_value: Optional[float]):
        """Set fixed w mode. Pass None to use adaptive gating."""
        self._fixed_w = w_value

    def compute_audio_features(self, audio_logits: torch.Tensor) -> torch.Tensor:
        """Extract features from audio logits for gating."""
        probs = F.softmax(audio_logits, dim=1)
        max_prob = probs.max(dim=1)[0]
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)

        top2_probs = torch.topk(probs, k=min(2, probs.size(1)), dim=1)[0]
        if top2_probs.size(1) >= 2:
            margin = top2_probs[:, 0] - top2_probs[:, 1]
        else:
            margin = top2_probs[:, 0]

        return torch.stack([max_prob, entropy, margin], dim=1)

    def compute_prior_features(self, prior_probs: torch.Tensor) -> torch.Tensor:
        """Extract features from prior probabilities for gating."""
        max_prob = prior_probs.max(dim=1)[0]
        entropy = -(prior_probs * torch.log(prior_probs + 1e-10)).sum(dim=1)

        top2_probs = torch.topk(prior_probs, k=min(2, prior_probs.size(1)), dim=1)[0]
        if top2_probs.size(1) >= 2:
            margin = top2_probs[:, 0] - top2_probs[:, 1]
        else:
            margin = top2_probs[:, 0]

        return torch.stack([max_prob, entropy, margin], dim=1)

    def forward(
        self,
        audio: torch.Tensor,
        prior_probs: torch.Tensor,
        metadata: Optional[List[Dict]] = None,
        dataset_names: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            final_logits, audio_logits, w (gating weights)
        """
        features = self.encoder(audio)
        audio_logits = self.classifier(features.float())

        temp = self.temperature.abs().clamp(min=1e-8)
        audio_logits_scaled = audio_logits / temp

        # Determine w value
        if self._fixed_w is not None:
            # Fixed w mode
            w = torch.full(
                (audio_logits.size(0),),
                self._fixed_w,
                device=audio_logits.device,
            )
        elif metadata is not None:
            # Adaptive w mode
            audio_features = self.compute_audio_features(audio_logits_scaled)
            prior_features = self.compute_prior_features(prior_probs)
            w = self.gate_network(audio_features, prior_features, metadata, dataset_names)
        else:
            # Fallback to init_w
            w = torch.full(
                (audio_logits.size(0),),
                self.gate_network.init_w,
                device=audio_logits.device,
            )

        min_prior = prior_probs.min()
        safe_eps = self.epsilon.clamp(min=-min_prior.item() + 1e-8)
        log_prior = torch.log(prior_probs + safe_eps)

        final_logits = audio_logits_scaled + w.unsqueeze(1) * log_prior

        return final_logits, audio_logits, w


# ============================================================================
# Prior Model
# ============================================================================


class PriorWrapper:
    """Wrapper for prior with cache support."""

    def __init__(
        self,
        species_list: List[str],
        cache_path: Optional[Path] = None,
    ):
        self.species_list = species_list
        self.num_species = len(species_list)
        self.use_cache = False

        if cache_path and Path(cache_path).exists():
            print(f"Loading prior cache from {cache_path}...")
            cache_path = Path(cache_path)

            if cache_path.suffix == '.h5':
                if not HAS_H5PY:
                    raise ImportError("h5py required for .h5 cache files")
                with h5py.File(cache_path, 'r') as f:
                    self.prior_matrix = f['priors'][:]
                    species_raw = f['species_list'][:]
                    self.cached_species = [
                        s.decode('utf-8') if isinstance(s, bytes) else s
                        for s in species_raw
                    ]
                    self.cached_indices = f['sample_indices'][:]
            else:
                npz = np.load(cache_path, allow_pickle=True)
                self.prior_matrix = npz['priors']
                self.cached_species = [
                    s.decode('utf-8') if isinstance(s, bytes) else s
                    for s in npz['species_list']
                ]
                self.cached_indices = npz['sample_indices']

            self.index_map = {idx: i for i, idx in enumerate(self.cached_indices)}
            self.use_cache = True
            print(f"Loaded cache: {len(self.cached_indices)} samples")

            # Reindex if species lists don't match
            if self.cached_species != self.species_list:
                self._reindex_species()
        else:
            print("No cache found, using uniform prior")

    def _reindex_species(self):
        """Reindex prior matrix to match current species list."""
        print(f"Reindexing priors: cache has {len(self.cached_species)}, "
              f"current has {len(self.species_list)} species")

        cached_species_to_idx = {s: i for i, s in enumerate(self.cached_species)}
        new_matrix = np.zeros(
            (self.prior_matrix.shape[0], self.num_species),
            dtype=self.prior_matrix.dtype
        )

        for i, species in enumerate(self.species_list):
            if species in cached_species_to_idx:
                new_matrix[:, i] = self.prior_matrix[:, cached_species_to_idx[species]]
            else:
                new_matrix[:, i] = 1.0 / self.num_species

        self.prior_matrix = new_matrix

        # Renormalize
        row_sums = self.prior_matrix.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        self.prior_matrix = self.prior_matrix / row_sums

    def get_batch(self, sample_indices: List[int]) -> np.ndarray:
        """Get prior probabilities for a batch."""
        batch_size = len(sample_indices)

        if not self.use_cache:
            return np.ones((batch_size, self.num_species), dtype=np.float32) / self.num_species

        priors = np.zeros((batch_size, self.num_species), dtype=np.float32)
        for i, idx in enumerate(sample_indices):
            if idx in self.index_map:
                priors[i] = self.prior_matrix[self.index_map[idx]]
            else:
                priors[i] = 1.0 / self.num_species

        return np.clip(priors, 1e-8, 1.0)


# ============================================================================
# Training Functions
# ============================================================================


def train_linear_probe(
    model: FusionModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    prior_model: PriorWrapper,
    device: str,
    epochs: int,
    lr: float,
    log_interval: int = 10,
    use_wandb: bool = False,
) -> float:
    """Train linear probe with frozen encoder (Phase 1)."""
    print("\n" + "=" * 60)
    print("PHASE 1: Training Linear Probe (shared across all models)")
    print("=" * 60)

    # Freeze everything except classifier
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")

    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=lr, weight_decay=0.01)
    scaler = GradScaler("cuda")

    best_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        model.encoder.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Phase 1 Epoch {epoch}/{epochs}")
        for batch_idx, batch in enumerate(pbar):
            audio = batch["audio"].to(device)
            labels = batch["label"].to(device)
            sample_indices = batch["sample_indices"]

            prior_probs = prior_model.get_batch(sample_indices)
            prior_probs = torch.from_numpy(prior_probs).float().to(device)

            optimizer.zero_grad()

            with autocast("cuda", dtype=torch.bfloat16):
                _, audio_logits, _ = model(audio, prior_probs, batch["metadata"])
                loss = F.cross_entropy(audio_logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * len(labels)
            preds = audio_logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

            if batch_idx % log_interval == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100*correct/total:.2f}%")

        train_loss = total_loss / total
        train_acc = correct / total

        # Validation
        val_metrics = evaluate_model(model, val_loader, prior_model, device, use_fusion=False)

        print(f"  Train Loss: {train_loss:.4f}, Acc: {100*train_acc:.2f}%")
        print(f"  Val   Loss: {val_metrics['loss']:.4f}, Acc: {100*val_metrics['accuracy']:.2f}%")

        if use_wandb and HAS_WANDB:
            wandb.log({
                "phase1/epoch": epoch,
                "phase1/train_loss": train_loss,
                "phase1/train_acc": train_acc,
                "phase1/val_loss": val_metrics['loss'],
                "phase1/val_acc": val_metrics['accuracy'],
            })

        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            best_state = {k: v.clone() for k, v in model.classifier.state_dict().items()}

    if best_state:
        model.classifier.load_state_dict(best_state)

    print(f"\nPhase 1 complete. Best probe accuracy: {100*best_acc:.2f}%")
    return best_acc


def train_adaptive_gating(
    model: FusionModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    prior_model: PriorWrapper,
    device: str,
    epochs: int,
    lr: float,
    variance_lambda: float = 0.01,
    log_interval: int = 10,
    use_wandb: bool = False,
) -> Tuple[float, Dict[str, Any]]:
    """Train adaptive gating network (Phase 2)."""
    print("\n" + "=" * 60)
    print("PHASE 2: Training Adaptive Gating Network")
    print("=" * 60)

    # Make sure we're in adaptive mode
    model.set_fixed_w(None)

    # Freeze classifier, train fusion parameters
    for param in model.parameters():
        param.requires_grad = False

    for param in model.gate_network.parameters():
        param.requires_grad = True
    model.temperature.requires_grad = True
    model.epsilon.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")

    fusion_params = list(model.gate_network.parameters()) + [model.temperature, model.epsilon]
    optimizer = torch.optim.AdamW(fusion_params, lr=lr, weight_decay=0.001)
    scaler = GradScaler("cuda")

    best_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        model.encoder.eval()

        total_loss = 0.0
        correct = 0
        total = 0
        all_w = []

        pbar = tqdm(train_loader, desc=f"Phase 2 Epoch {epoch}/{epochs}")
        for batch_idx, batch in enumerate(pbar):
            audio = batch["audio"].to(device)
            labels = batch["label"].to(device)
            sample_indices = batch["sample_indices"]
            metadata = batch["metadata"]

            prior_probs = prior_model.get_batch(sample_indices)
            prior_probs = torch.from_numpy(prior_probs).float().to(device)

            optimizer.zero_grad()

            with autocast("cuda", dtype=torch.bfloat16):
                final_logits, _, w = model(audio, prior_probs, metadata)
                ce_loss = F.cross_entropy(final_logits, labels)

                w_float = w.float()
                w_var = w_float.var()
                var_loss = -variance_lambda * w_var

                loss = ce_loss + var_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * len(labels)
            preds = final_logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)
            all_w.extend(w.detach().cpu().float().numpy().tolist())

            if batch_idx % log_interval == 0:
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    acc=f"{100*correct/total:.2f}%",
                    w_mean=f"{w.mean().item():.4f}"
                )

        train_loss = total_loss / total
        train_acc = correct / total
        all_w = np.array(all_w)

        # Validation
        val_metrics = evaluate_model(model, val_loader, prior_model, device, use_fusion=True)

        print(f"  Train Loss: {train_loss:.4f}, Acc: {100*train_acc:.2f}%")
        print(f"  Val   Loss: {val_metrics['loss']:.4f}, Acc: {100*val_metrics['accuracy']:.2f}%")
        print(f"  Gating w: mean={all_w.mean():.4f}, std={all_w.std():.4f}")
        print(f"  Temperature={model.temperature.item():.4f}, Epsilon={model.epsilon.item():.6f}")

        if use_wandb and HAS_WANDB:
            wandb.log({
                "adaptive/epoch": epoch,
                "adaptive/train_loss": train_loss,
                "adaptive/train_acc": train_acc,
                "adaptive/val_loss": val_metrics['loss'],
                "adaptive/val_acc": val_metrics['accuracy'],
                "adaptive/w_mean": all_w.mean(),
                "adaptive/w_std": all_w.std(),
                "adaptive/temperature": model.temperature.item(),
                "adaptive/epsilon": model.epsilon.item(),
            })

        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            best_state = {
                "gate_network": {k: v.clone() for k, v in model.gate_network.state_dict().items()},
                "temperature": model.temperature.data.clone(),
                "epsilon": model.epsilon.data.clone(),
            }

    if best_state:
        model.gate_network.load_state_dict(best_state["gate_network"])
        model.temperature.data = best_state["temperature"]
        model.epsilon.data = best_state["epsilon"]

    # Final evaluation
    final_metrics = evaluate_model(model, val_loader, prior_model, device, use_fusion=True)

    print(f"\nAdaptive gating training complete. Best accuracy: {100*best_acc:.2f}%")
    return best_acc, final_metrics


@torch.no_grad()
def evaluate_model(
    model: FusionModel,
    dataloader: DataLoader,
    prior_model: PriorWrapper,
    device: str,
    use_fusion: bool = True,
) -> Dict[str, Any]:
    """Evaluate model on dataset."""
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_logits = []
    all_w = []

    for batch in dataloader:
        audio = batch["audio"].to(device)
        labels = batch["label"].to(device)
        sample_indices = batch["sample_indices"]
        metadata = batch["metadata"]

        prior_probs = prior_model.get_batch(sample_indices)
        prior_probs = torch.from_numpy(prior_probs).float().to(device)

        with autocast("cuda", dtype=torch.bfloat16):
            final_logits, audio_logits, w = model(audio, prior_probs, metadata)
            logits = final_logits if use_fusion else audio_logits
            loss = F.cross_entropy(logits, labels)

        total_loss += loss.item() * len(labels)
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_logits.append(logits.float().cpu())
        all_w.extend(w.float().cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_w = np.array(all_w)

    # Compute metrics
    metrics = {
        "loss": total_loss / len(all_labels),
        "accuracy": accuracy_score(all_labels, all_preds) if HAS_SKLEARN else (all_preds == all_labels).mean(),
        "w_mean": float(all_w.mean()),
        "w_std": float(all_w.std()),
        "num_samples": len(all_labels),
    }

    # Additional metrics
    if HAS_SKLEARN:
        num_classes = all_logits.shape[1]
        probs = torch.softmax(torch.from_numpy(all_logits), dim=1).numpy()

        try:
            labels_onehot = np.eye(num_classes)[all_labels]
            metrics["mAP"] = average_precision_score(labels_onehot, probs, average="macro")
        except Exception:
            pass

        try:
            metrics["top5_accuracy"] = top_k_accuracy_score(
                all_labels, all_logits, k=min(5, num_classes)
            )
        except Exception:
            pass

    return metrics


# ============================================================================
# Dataset Loading
# ============================================================================


def create_subset_dataloaders(
    data_root: Path,
    subset_size: int,
    batch_size: int,
    val_split: float,
    num_workers: int,
    seed: int,
) -> Tuple[DataLoader, DataLoader, Dict[str, int], List[str], PriorWrapper]:
    """
    Create train and validation dataloaders for a subset of CBI.

    Returns:
        train_loader, val_loader, label_to_idx, species_list, prior_model
    """
    # Load full CBI dataset
    data_dir = data_root / "beans" / "cbi"
    full_dataset = CBIDataset(str(data_dir), split="train")

    # Build label mapping
    species_list = full_dataset.get_all_labels()
    label_to_idx = {s: i for i, s in enumerate(species_list)}
    full_dataset.label_to_idx = label_to_idx

    print(f"Total CBI samples: {len(full_dataset)}")
    print(f"Total classes: {len(species_list)}")

    # Create subset
    generator = torch.Generator().manual_seed(seed)
    subset_size = min(subset_size, len(full_dataset))

    indices = torch.randperm(len(full_dataset), generator=generator)[:subset_size].tolist()
    subset = Subset(full_dataset, indices)

    print(f"Subset size: {len(subset)}")

    # Split into train/val
    val_size = int(len(subset) * val_split)
    train_size = len(subset) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        subset, [train_size, val_size], generator=generator
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Load prior cache
    cache_path = CBIDataset.get_priors_cache_path(data_dir)
    prior_model = PriorWrapper(species_list, cache_path)

    return train_loader, val_loader, label_to_idx, species_list, prior_model


# ============================================================================
# Main Sweep
# ============================================================================


def run_sweep(
    model: FusionModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    prior_model: PriorWrapper,
    device: str,
    args,
    use_wandb: bool = False,
) -> Dict[str, Any]:
    """
    Run the full w-sweep experiment.

    Returns:
        Dictionary with results for all models
    """
    results = {
        "fixed_w": {},
        "adaptive": {},
        "probe_accuracy": None,
    }

    # Check if we're in phase1_only mode
    if args.phase1_only:
        # Phase 1: Train shared linear probe and save checkpoint
        probe_acc = train_linear_probe(
            model, train_loader, val_loader, prior_model, device,
            epochs=args.phase1_epochs,
            lr=args.phase1_lr,
            log_interval=args.log_interval,
            use_wandb=use_wandb,
        )
        results["probe_accuracy"] = probe_acc

        # Save probe checkpoint
        probe_state = {k: v.clone() for k, v in model.classifier.state_dict().items()}
        initial_temp = model.temperature.data.clone()
        initial_eps = model.epsilon.data.clone()

        probe_checkpoint_path = args.exp_dir / "probe_checkpoint.pt"
        torch.save({
            "classifier": probe_state,
            "temperature": initial_temp,
            "epsilon": initial_eps,
            "probe_accuracy": probe_acc,
        }, probe_checkpoint_path)
        print(f"\n[PHASE1-ONLY MODE] Saved probe checkpoint to: {probe_checkpoint_path}")
        print(f"Probe accuracy: {100*probe_acc:.2f}%")
        print("\nTo run adaptive training, use:")
        print(f"  python w_sweep.py --adaptive_only --probe_checkpoint {probe_checkpoint_path}")

        return results

    # Check if we're in adaptive_only mode
    elif args.adaptive_only:
        # Load probe checkpoint
        if args.probe_checkpoint is None:
            raise ValueError("--probe_checkpoint is required when using --adaptive_only")

        probe_path = Path(args.probe_checkpoint)
        if not probe_path.exists():
            raise FileNotFoundError(f"Probe checkpoint not found: {probe_path}")

        print(f"\n[ADAPTIVE-ONLY MODE] Loading probe from: {probe_path}")
        checkpoint = torch.load(probe_path, map_location=device, weights_only=True)
        model.classifier.load_state_dict(checkpoint["classifier"])
        model.temperature.data = checkpoint["temperature"].to(device)
        model.epsilon.data = checkpoint["epsilon"].to(device)
        results["probe_accuracy"] = checkpoint.get("probe_accuracy", None)
        print(f"Loaded probe with accuracy: {100*results['probe_accuracy']:.2f}%" if results['probe_accuracy'] else "Loaded probe checkpoint")

        # Skip to adaptive training
        probe_state = {k: v.clone() for k, v in model.classifier.state_dict().items()}
        initial_temp = model.temperature.data.clone()
        initial_eps = model.epsilon.data.clone()

    else:
        # Phase 1: Train shared linear probe
        probe_acc = train_linear_probe(
            model, train_loader, val_loader, prior_model, device,
            epochs=args.phase1_epochs,
            lr=args.phase1_lr,
            log_interval=args.log_interval,
            use_wandb=use_wandb,
        )
        results["probe_accuracy"] = probe_acc

        # Save probe state for reuse
        probe_state = {k: v.clone() for k, v in model.classifier.state_dict().items()}
        initial_temp = model.temperature.data.clone()
        initial_eps = model.epsilon.data.clone()

        # Save probe checkpoint for future runs
        probe_checkpoint_path = args.exp_dir / "probe_checkpoint.pt"
        torch.save({
            "classifier": probe_state,
            "temperature": initial_temp,
            "epsilon": initial_eps,
            "probe_accuracy": probe_acc,
        }, probe_checkpoint_path)
        print(f"Saved probe checkpoint to: {probe_checkpoint_path}")

        # Evaluate fixed-w models
        print("\n" + "=" * 60)
        print("FIXED-W SWEEP: Evaluating models with fixed gating values")
        print("=" * 60)

        for w_value in FIXED_W_VALUES:
            print(f"\n--- Evaluating w = {w_value} ---")

            # Reset to probe state
            model.classifier.load_state_dict(probe_state)
            model.temperature.data = initial_temp.clone()
            model.epsilon.data = initial_eps.clone()

            # Set fixed w
            model.set_fixed_w(w_value)

            # Evaluate
            metrics = evaluate_model(model, val_loader, prior_model, device, use_fusion=True)

            results["fixed_w"][str(w_value)] = metrics
            print(f"  Accuracy: {100*metrics['accuracy']:.2f}%")
            if "mAP" in metrics:
                print(f"  mAP: {100*metrics['mAP']:.2f}%")

            if use_wandb and HAS_WANDB:
                wandb.log({
                    f"fixed_w_{w_value}/accuracy": metrics['accuracy'],
                    f"fixed_w_{w_value}/loss": metrics['loss'],
                    f"fixed_w_{w_value}/mAP": metrics.get('mAP', 0),
                })

    # Train and evaluate adaptive model
    print("\n" + "=" * 60)
    print("ADAPTIVE MODEL: Training gating network")
    print("=" * 60)

    # Reset to probe state
    model.classifier.load_state_dict(probe_state)
    model.temperature.data = initial_temp.clone()
    model.epsilon.data = initial_eps.clone()

    # Reset gating network
    model.gate_network = PriorGatingNetwork(
        input_dim=13,
        hidden_dim=args.gate_hidden_dim,
        w_max=args.w_max,
        init_w=args.init_w,
    ).to(device)

    adaptive_acc, adaptive_metrics = train_adaptive_gating(
        model, train_loader, val_loader, prior_model, device,
        epochs=args.phase2_epochs,
        lr=args.phase2_lr,
        variance_lambda=args.variance_lambda,
        log_interval=args.log_interval,
        use_wandb=use_wandb,
    )

    results["adaptive"] = {
        "best_accuracy": adaptive_acc,
        "final_metrics": adaptive_metrics,
        "temperature": model.temperature.item(),
        "epsilon": model.epsilon.item(),
    }

    return results


def print_sweep_summary(results: Dict[str, Any]):
    """Print a summary table of sweep results."""
    print("\n")
    print("=" * 70)
    print("                     W-SWEEP RESULTS SUMMARY")
    print("=" * 70)

    if results["probe_accuracy"] is not None:
        print("\n  Shared Probe Accuracy: {:.2f}%".format(100 * results["probe_accuracy"]))

    # Fixed-W results (may be empty in adaptive_only mode)
    if results["fixed_w"]:
        print("\n  Fixed-W Models:")
        print("  " + "-" * 50)
        print("  {:>8}  {:>12}  {:>12}  {:>12}".format("w", "Accuracy", "mAP", "Loss"))
        print("  " + "-" * 50)

        for w_str, metrics in sorted(results["fixed_w"].items(), key=lambda x: float(x[0])):
            acc = 100 * metrics["accuracy"]
            mAP = 100 * metrics.get("mAP", 0)
            loss = metrics["loss"]
            print(f"  {float(w_str):>8.2f}  {acc:>11.2f}%  {mAP:>11.2f}%  {loss:>12.4f}")

        print("  " + "-" * 50)
    else:
        print("\n  [Adaptive-only mode: Fixed-W results skipped]")

    # Adaptive results (may be empty in phase1_only mode)
    if results["adaptive"]:
        print("\n  Adaptive Model:")
        adaptive = results["adaptive"]
        print(f"    Best Accuracy: {100 * adaptive['best_accuracy']:.2f}%")
        if "final_metrics" in adaptive:
            fm = adaptive["final_metrics"]
            print(f"    Final Accuracy: {100 * fm['accuracy']:.2f}%")
            if "mAP" in fm:
                print(f"    Final mAP: {100 * fm['mAP']:.2f}%")
            print(f"    w_mean: {fm['w_mean']:.4f}, w_std: {fm['w_std']:.4f}")
        print(f"    Temperature: {adaptive['temperature']:.4f}")
        print(f"    Epsilon: {adaptive['epsilon']:.6f}")
    else:
        print("\n  [Phase1-only mode: Adaptive results not yet computed]")

    # Comparison only if we have both fixed-w and adaptive results
    if results["fixed_w"] and results["adaptive"]:
        best_fixed_w = max(results["fixed_w"].items(), key=lambda x: x[1]["accuracy"])
        print(f"\n  Best Fixed-W: w={best_fixed_w[0]} with {100*best_fixed_w[1]['accuracy']:.2f}% accuracy")

        adaptive_acc = results["adaptive"]["final_metrics"]["accuracy"]
        best_fixed_acc = best_fixed_w[1]["accuracy"]
        diff = (adaptive_acc - best_fixed_acc) * 100
        if diff > 0:
            print(f"  Adaptive vs Best Fixed: +{diff:.2f}% (adaptive wins)")
        else:
            print(f"  Adaptive vs Best Fixed: {diff:.2f}% (fixed w={best_fixed_w[0]} wins)")

    print("=" * 70)


def parse_args():
    parser = argparse.ArgumentParser(
        description="W-sweep experiment for bio-acoustic classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument("--data_root", type=str, default=None, help="Data root directory")
    parser.add_argument("--subset_size", type=int, default=1000, help="Number of samples to use")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")

    # Model
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "max", "cls"])
    parser.add_argument("--w_max", type=float, default=2.0)
    parser.add_argument("--init_w", type=float, default=0.0308)
    parser.add_argument("--init_temperature", type=float, default=0.5101)
    parser.add_argument("--init_epsilon", type=float, default=-0.049955)
    parser.add_argument("--gate_hidden_dim", type=int, default=64)

    # Training
    parser.add_argument("--phase1_epochs", type=int, default=10, help="Epochs for linear probe")
    parser.add_argument("--phase2_epochs", type=int, default=10, help="Epochs for adaptive gating")
    parser.add_argument("--phase1_lr", type=float, default=1e-3)
    parser.add_argument("--phase2_lr", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--variance_lambda", type=float, default=0.01)

    # Hardware
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    # Logging
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--log_interval", type=int, default=10)

    # Output
    parser.add_argument("--exp_dir", type=str, default=None, help="Experiment directory")

    # Resume options
    parser.add_argument("--phase1_only", action="store_true",
                        help="Only run Phase 1 (probe training); save checkpoint and exit")
    parser.add_argument("--adaptive_only", action="store_true",
                        help="Skip Phase 1 and fixed-w sweep; only train adaptive gating")
    parser.add_argument("--probe_checkpoint", type=str, default=None,
                        help="Path to probe checkpoint (.pt) to load for adaptive_only mode")

    return parser.parse_args()


def main():
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Setup paths
    if args.data_root is None:
        args.data_root = PROJECT_ROOT / "Data"
    else:
        args.data_root = Path(args.data_root)

    # Create experiment directory
    if args.exp_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.exp_dir = PROJECT_ROOT / "experiments" / f"w_sweep_{timestamp}"
    else:
        args.exp_dir = Path(args.exp_dir)

    args.exp_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args).copy()
    config["data_root"] = str(config["data_root"])
    config["exp_dir"] = str(config["exp_dir"])
    config["fixed_w_values"] = FIXED_W_VALUES
    with open(args.exp_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Device
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create dataloaders
    train_loader, val_loader, label_to_idx, species_list, prior_model = create_subset_dataloaders(
        data_root=args.data_root,
        subset_size=args.subset_size,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    num_classes = len(species_list)

    # Initialize model
    encoder = FrozenAudioEncoder(pooling=args.pooling)
    model = FusionModel(
        encoder=encoder,
        num_classes=num_classes,
        w_max=args.w_max,
        init_w=args.init_w,
        init_temperature=args.init_temperature,
        init_epsilon=args.init_epsilon,
        gate_hidden_dim=args.gate_hidden_dim,
    ).to(device)

    # Initialize wandb
    use_wandb = HAS_WANDB and args.wandb_project is not None
    if use_wandb:
        run_name = f"w_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(project=args.wandb_project, name=run_name, config=config)

    # Run sweep
    results = run_sweep(
        model, train_loader, val_loader, prior_model, device, args,
        use_wandb=use_wandb,
    )

    # Save results
    results_path = args.exp_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    print_sweep_summary(results)

    print(f"\nResults saved to: {args.exp_dir}")

    if use_wandb:
        # Log summary metrics
        summary = {
            "probe_accuracy": results["probe_accuracy"],
            "adaptive_accuracy": results["adaptive"]["final_metrics"]["accuracy"],
        }
        for w_str, metrics in results["fixed_w"].items():
            summary[f"fixed_w_{w_str}_accuracy"] = metrics["accuracy"]

        wandb.log({"summary": summary})
        wandb.finish()


if __name__ == "__main__":
    main()
