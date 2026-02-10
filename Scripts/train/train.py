#!/usr/bin/env python3
"""
Unified training script for bio-acoustic classification.

Supports BEANS (CBI) and Birdset (PER, NES, UHH, HSN, SSW, SNE) datasets.

Two-phase training:
    Phase 1: Train linear probe on frozen encoder features
    Phase 2: Freeze probe, train gating network and fusion parameters

Model: final_logits = audio_logits / T + w(a,x,t) * log(prior + eps)

Usage:
    # Single dataset
    python train.py --dataset cbi

    # Multiple datasets
    python train.py --dataset PER,NES,UHH

    # All datasets
    python train.py --dataset all
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
from torch.utils.data import DataLoader, ConcatDataset, random_split
from tqdm import tqdm

# Add parent directories to path
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "NatureLM-audio"))
sys.path.insert(0, str(SCRIPT_DIR))

from datasets import CBIDataset, BirdsetDataset, collate_fn, BIRDSET_DATASETS, BEANS_DATASETS

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
        roc_auc_score,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


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


def get_rng_state() -> Dict[str, Any]:
    """Get current RNG state for checkpointing."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_rng_state(state: Dict[str, Any]):
    """Restore RNG state from checkpoint."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


# ============================================================================
# Models
# ============================================================================


class AudioEncoder(nn.Module):
    """NatureLM audio encoder with optional fine-tuning support."""

    def __init__(self, pooling: str = "mean", trainable: bool = False):
        super().__init__()
        self.pooling = pooling
        self.trainable = trainable

        print("Loading NatureLM-audio encoder...")
        from NatureLM.models import NatureLM
        self.naturelm = NatureLM.from_pretrained("EarthSpeciesProject/NatureLM-audio")

        # Set parameter training mode
        for param in self.naturelm.parameters():
            param.requires_grad = trainable

        if not trainable:
            self.naturelm.eval()

        # Get encoder dimension
        self.encoder_dim = self.naturelm.llama_model.config.hidden_size
        print(f"Encoder dimension: {self.encoder_dim}")
        print(f"Encoder trainable: {trainable}")

    def set_trainable(self, trainable: bool):
        """Enable or disable gradient computation for encoder parameters."""
        self.trainable = trainable
        for param in self.naturelm.parameters():
            param.requires_grad = trainable
        if trainable:
            print("Encoder unfrozen - gradients enabled")
        else:
            print("Encoder frozen - gradients disabled")

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract pooled features from audio."""
        device = next(self.naturelm.parameters()).device
        audio = audio.to(device)

        # Use no_grad context if not trainable for efficiency
        if self.trainable:
            # Training mode - allow gradients
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                audio_embeds, audio_atts = self.naturelm.encode_audio(audio)
        else:
            # Inference mode - no gradients
            self.naturelm.eval()
            with torch.no_grad():
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


# Alias for backward compatibility
FrozenAudioEncoder = AudioEncoder


def get_dataset_type_indicator(dataset_names: List[str]) -> List[float]:
    """
    Get dataset type indicator for each sample.

    Returns:
        List of floats: 0.0 for BEANS (CBI), 1.0 for Birdset
    """
    indicators = []
    for name in dataset_names:
        if name.lower() in BEANS_DATASETS:
            indicators.append(0.0)
        elif name.upper() in BIRDSET_DATASETS:
            indicators.append(1.0)
        else:
            indicators.append(0.5)  # Unknown
    return indicators


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

        # Dataset type indicator (0 = BEANS/local time, 1 = Birdset/Zulu time)
        if dataset_names is not None:
            ds_type = torch.tensor(
                get_dataset_type_indicator(dataset_names),
                dtype=torch.float32,
                device=device,
            )
        else:
            ds_type = torch.zeros(len(metadata), dtype=torch.float32, device=device)

        meta_features = torch.stack(
            [day_sin, day_cos, hour_sin, hour_cos, lat_norm, lon_norm, ds_type], dim=1
        )
        combined = torch.cat([audio_features, prior_features, meta_features], dim=1)

        gate_logit = self.mlp(combined).squeeze(-1)
        w = torch.sigmoid(gate_logit) * self.w_max

        return w


class SpatioTemporalPriorModel(nn.Module):
    """
    Trainable spatio-temporal prior model.

    Learns to predict class distributions from spatio-temporal context:
        f(day_of_year, hour, latitude, longitude, dataset_type) -> p(y)

    Input features (7-dim):
        - sin(day), cos(day): Cyclical day encoding
        - sin(hour), cos(hour): Cyclical hour encoding
        - latitude (normalized), longitude (normalized)
        - dataset_type: 0 for BEANS/CBI (local time), 1 for Birdset (UTC/Zulu time)

    Output: Softmax probability distribution over classes
    """

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes

        # 7 input features: day_sin, day_cos, hour_sin, hour_cos, lat, lon, dataset_type
        self.mlp = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # Initialize to predict uniform distribution
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def _encode_metadata(
        self,
        metadata: List[Dict],
        device: torch.device,
        dataset_names: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """Encode metadata into feature tensor."""

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

        # Dataset type indicator (0 = BEANS/local time, 1 = Birdset/Zulu time)
        if dataset_names is not None:
            ds_type = torch.tensor(
                get_dataset_type_indicator(dataset_names),
                dtype=torch.float32,
                device=device,
            )
        else:
            ds_type = torch.zeros(len(metadata), dtype=torch.float32, device=device)

        features = torch.stack(
            [day_sin, day_cos, hour_sin, hour_cos, lat_norm, lon_norm, ds_type], dim=1
        )
        return features

    def forward(
        self,
        metadata: List[Dict],
        dataset_names: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Predict class probabilities from metadata.

        Args:
            metadata: List of metadata dicts with day_of_year, hour, latitude, longitude
            dataset_names: Optional list of dataset names for dataset-type-aware encoding

        Returns:
            Tensor of shape (batch_size, num_classes) - probability distribution
        """
        device = next(self.parameters()).device
        features = self._encode_metadata(metadata, device, dataset_names)
        logits = self.mlp(features)
        return F.softmax(logits, dim=1)

    def get_log_probs(
        self,
        metadata: List[Dict],
        dataset_names: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """Get log probabilities for use in fusion."""
        device = next(self.parameters()).device
        features = self._encode_metadata(metadata, device, dataset_names)
        logits = self.mlp(features)
        return F.log_softmax(logits, dim=1)


class FusionModel(nn.Module):
    """
    Fusion model with gating network.

    final_logits = audio_logits / temperature + w(a,x,t) * log(prior + epsilon)
    """

    def __init__(
        self,
        encoder: AudioEncoder,
        num_classes: int,
        w_max: float = 2.0,
        init_w: float = 0.0308,
        init_temperature: float = 0.5101,
        init_epsilon: float = -0.049955,
        gate_hidden_dim: int = 64,
    ):
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes

        classifier_hidden_dim: int = 256
        classifier_dropout: float = 0.3
        classifier_num_layers: int = 2
        
        # Linear probe (classifier head)
        # self.classifier = nn.Linear(encoder.encoder_dim, num_classes)
        self.classifier = self._build_classifier_mlp(
            input_dim=encoder.encoder_dim,
            hidden_dim=classifier_hidden_dim,
            output_dim=num_classes,
            num_layers=classifier_num_layers,
            dropout=classifier_dropout,
        )

        # Learnable fusion parameters
        self.temperature = nn.Parameter(torch.tensor(init_temperature))
        self.epsilon = nn.Parameter(torch.tensor(init_epsilon))

        # Gating network
        self.gate_network = PriorGatingNetwork(
            input_dim=13,
            hidden_dim=gate_hidden_dim,
            w_max=w_max,
            init_w=init_w,
        )
        
    def _build_classifier_mlp(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> nn.Sequential:
        """Build MLP classifier with regularization (LayerNorm, Dropout)."""
        layers = []
        
        # Input layer
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ])
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        
        # Output layer (no activation, no dropout)
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        return nn.Sequential(*layers)

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

        Args:
            audio: Audio tensor
            prior_probs: Prior probability tensor
            metadata: List of metadata dicts
            dataset_names: List of dataset names for dataset-type-aware gating

        Returns:
            final_logits, audio_logits, w (gating weights)
        """
        features = self.encoder(audio)
        audio_logits = self.classifier(features.float())

        temp = self.temperature.abs().clamp(min=1e-8)
        audio_logits_scaled = audio_logits / temp

        if metadata is not None:
            audio_features = self.compute_audio_features(audio_logits_scaled)
            prior_features = self.compute_prior_features(prior_probs)
            w = self.gate_network(audio_features, prior_features, metadata, dataset_names)
        else:
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


def train_spatio_temporal_prior(
    prior_model: SpatioTemporalPriorModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int = 20,
    lr: float = 1e-3,
    log_interval: int = 10,
    use_wandb: bool = False,
) -> float:
    """
    Train the spatio-temporal prior model.

    This model learns to predict class distributions from metadata alone,
    providing a learned prior for fusion with the audio model.

    Args:
        prior_model: SpatioTemporalPriorModel instance
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        epochs: Number of training epochs
        lr: Learning rate
        log_interval: How often to log progress
        use_wandb: Whether to log to W&B

    Returns:
        Best validation accuracy
    """
    print("\n" + "=" * 60)
    print("PRIOR PHASE: Training Spatio-Temporal Prior Model")
    print("=" * 60)

    optimizer = torch.optim.AdamW(prior_model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        prior_model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Prior Epoch {epoch}/{epochs}")
        for batch_idx, batch in enumerate(pbar):
            labels = batch["label"].to(device)
            metadata = batch["metadata"]
            dataset_names = batch.get("dataset_names", None)

            optimizer.zero_grad()

            probs = prior_model(metadata, dataset_names)
            loss = F.cross_entropy(probs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)
            preds = probs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

            if batch_idx % log_interval == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100*correct/total:.2f}%")

        scheduler.step()
        train_loss = total_loss / total
        train_acc = correct / total

        # Validation
        prior_model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                labels = batch["label"].to(device)
                metadata = batch["metadata"]
                dataset_names = batch.get("dataset_names", None)
                probs = prior_model(metadata, dataset_names)
                preds = probs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)

        val_acc = val_correct / val_total

        print(f"  Train Loss: {train_loss:.4f}, Acc: {100*train_acc:.2f}%")
        print(f"  Val   Acc: {100*val_acc:.2f}%")

        if use_wandb and HAS_WANDB:
            wandb.log({
                "prior/epoch": epoch,
                "prior/train_loss": train_loss,
                "prior/train_acc": train_acc,
                "prior/val_acc": val_acc,
            })

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in prior_model.state_dict().items()}

    if best_state:
        prior_model.load_state_dict(best_state)

    print(f"\nPrior training complete. Best accuracy: {100*best_acc:.2f}%")
    return best_acc


def train_phase1(
    model: FusionModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    prior_model: PriorWrapper,
    device: str,
    epochs: int,
    lr: float,
    log_interval: int = 10,
    use_wandb: bool = False,
    finetune_encoder: bool = False,
    encoder_lr_factor: float = 0.1,
) -> float:
    """Phase 1: Train linear probe, optionally with encoder fine-tuning.

    Args:
        model: FusionModel instance
        train_loader: Training data loader
        val_loader: Validation data loader
        prior_model: Prior probability wrapper
        device: Device to train on
        epochs: Number of training epochs
        lr: Learning rate for classifier head
        log_interval: How often to log progress
        use_wandb: Whether to log to W&B
        finetune_encoder: If True, unfreeze and fine-tune the audio encoder
        encoder_lr_factor: Learning rate multiplier for encoder (default 0.1 = 10x lower)
    """
    print("\n" + "=" * 60)
    print("PHASE 1: Training Linear Probe" + (" + Encoder Fine-tuning" if finetune_encoder else ""))
    print("=" * 60)

    # Freeze everything except classifier (and optionally encoder)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    # Optionally unfreeze encoder for fine-tuning
    if finetune_encoder:
        model.encoder.set_trainable(True)
        encoder_lr = lr * encoder_lr_factor
        print(f"Encoder learning rate: {encoder_lr:.2e} ({encoder_lr_factor}x classifier LR)")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")

    # Create optimizer with separate parameter groups
    if finetune_encoder:
        param_groups = [
            {"params": model.classifier.parameters(), "lr": lr},
            {"params": model.encoder.naturelm.parameters(), "lr": lr * encoder_lr_factor},
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=lr, weight_decay=0.01)

    scaler = GradScaler("cuda")

    best_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        # Set encoder mode based on fine-tuning flag
        if finetune_encoder:
            model.encoder.naturelm.train()
        else:
            model.encoder.naturelm.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Phase 1 Epoch {epoch}/{epochs}")
        for batch_idx, batch in enumerate(pbar):
            audio = batch["audio"].to(device)
            labels = batch["label"].to(device)
            sample_indices = batch["sample_indices"]
            dataset_names = batch.get("dataset_names", None)

            prior_probs = prior_model.get_batch(sample_indices)
            prior_probs = torch.from_numpy(prior_probs).float().to(device)

            optimizer.zero_grad()

            with autocast("cuda", dtype=torch.bfloat16):
                _, audio_logits, _ = model(audio, prior_probs, batch["metadata"], dataset_names)
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
        val_metrics = evaluate(model, val_loader, prior_model, device, use_fusion=False)

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


def train_phase2(
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
) -> float:
    """Phase 2: Freeze probe, train gating network and fusion parameters."""
    print("\n" + "=" * 60)
    print("PHASE 2: Training Gating Network")
    print("=" * 60)
    print(f"  Variance regularization lambda: {variance_lambda}")

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
        total_ce_loss = 0.0
        total_var_loss = 0.0
        correct = 0
        total = 0
        all_w = []

        pbar = tqdm(train_loader, desc=f"Phase 2 Epoch {epoch}/{epochs}")
        for batch_idx, batch in enumerate(pbar):
            audio = batch["audio"].to(device)
            labels = batch["label"].to(device)
            sample_indices = batch["sample_indices"]
            metadata = batch["metadata"]
            dataset_names = batch.get("dataset_names", None)

            prior_probs = prior_model.get_batch(sample_indices)
            prior_probs = torch.from_numpy(prior_probs).float().to(device)

            optimizer.zero_grad()

            with autocast("cuda", dtype=torch.bfloat16):
                final_logits, _, w = model(audio, prior_probs, metadata, dataset_names)
                ce_loss = F.cross_entropy(final_logits, labels)

                w_float = w.float()
                w_var = w_float.var()
                var_loss = -variance_lambda * w_var

                loss = ce_loss + var_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * len(labels)
            total_ce_loss += ce_loss.item() * len(labels)
            total_var_loss += var_loss.item() * len(labels)
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
        val_metrics = evaluate(model, val_loader, prior_model, device, use_fusion=True)

        print(f"  Train Loss: {train_loss:.4f}, Acc: {100*train_acc:.2f}%")
        print(f"  Val   Loss: {val_metrics['loss']:.4f}, Acc: {100*val_metrics['accuracy']:.2f}%")
        print(f"  Gating w: mean={all_w.mean():.4f}, std={all_w.std():.4f}")
        print(f"  Temperature={model.temperature.item():.4f}, Epsilon={model.epsilon.item():.6f}")

        if use_wandb and HAS_WANDB:
            wandb.log({
                "phase2/epoch": epoch,
                "phase2/train_loss": train_loss,
                "phase2/train_acc": train_acc,
                "phase2/val_loss": val_metrics['loss'],
                "phase2/val_acc": val_metrics['accuracy'],
                "phase2/w_mean": all_w.mean(),
                "phase2/w_std": all_w.std(),
                "phase2/temperature": model.temperature.item(),
                "phase2/epsilon": model.epsilon.item(),
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

    print(f"\nPhase 2 complete. Best fused accuracy: {100*best_acc:.2f}%")
    return best_acc


def compute_metrics_for_dataset_type(
    all_labels: np.ndarray,
    all_multilabels: np.ndarray,
    all_logits: np.ndarray,
    all_preds: np.ndarray,
    dataset_type: str,
) -> Dict[str, float]:
    """
    Compute dataset-specific metrics.

    Args:
        all_labels: Single-label targets (N,)
        all_multilabels: Multi-hot targets (N, C)
        all_logits: Model logits (N, C)
        all_preds: Model predictions (N,)
        dataset_type: "beans" or "birdset"

    Returns:
        Dictionary of computed metrics
    """
    metrics = {}

    if not HAS_SKLEARN:
        metrics["accuracy"] = (all_preds == all_labels).mean()
        return metrics

    num_classes = all_logits.shape[1]
    probs = torch.softmax(torch.from_numpy(all_logits), dim=1).numpy()

    # Accuracy (common to both)
    metrics["accuracy"] = accuracy_score(all_labels, all_preds)

    if dataset_type == "beans":
        # BEANS metrics: accuracy, mAP
        try:
            labels_onehot = np.eye(num_classes)[all_labels]
            metrics["mAP"] = average_precision_score(
                labels_onehot, probs, average="macro"
            )
        except Exception:
            pass

        try:
            metrics["top5_accuracy"] = top_k_accuracy_score(
                all_labels, all_logits, k=min(5, num_classes)
            )
        except Exception:
            pass

    elif dataset_type == "birdset":
        # BirdSet metrics: accuracy, ROC-AUC, cmAP (all using multi-label targets)

        # Use multi-label targets for ROC-AUC and cmAP
        multilabel_targets = all_multilabels

        # Filter out classes with no positive samples for stable metrics
        class_has_positive = multilabel_targets.sum(axis=0) > 0
        class_has_negative = (1 - multilabel_targets).sum(axis=0) > 0
        valid_classes = class_has_positive & class_has_negative

        if valid_classes.sum() > 0:
            valid_targets = multilabel_targets[:, valid_classes]
            valid_probs = probs[:, valid_classes]

            # ROC-AUC (macro average across classes)
            try:
                metrics["roc_auc"] = roc_auc_score(
                    valid_targets, valid_probs, average="macro"
                )
            except Exception as e:
                print(f"Warning: Could not compute ROC-AUC: {e}")

            # cmAP (class mean Average Precision)
            try:
                metrics["cmAP"] = average_precision_score(
                    valid_targets, valid_probs, average="macro"
                )
            except Exception as e:
                print(f"Warning: Could not compute cmAP: {e}")

        try:
            metrics["top5_accuracy"] = top_k_accuracy_score(
                all_labels, all_logits, k=min(5, num_classes)
            )
        except Exception:
            pass

    return metrics


@torch.no_grad()
def evaluate(
    model: FusionModel,
    dataloader: DataLoader,
    prior_model: PriorWrapper,
    device: str,
    use_fusion: bool = True,
    compute_per_dataset: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate model on dataset.

    Args:
        model: The fusion model to evaluate
        dataloader: Data loader for evaluation
        prior_model: Prior probability wrapper
        device: Device to evaluate on
        use_fusion: Whether to use fused logits (True) or audio-only (False)
        compute_per_dataset: Whether to compute per-dataset metrics

    Returns:
        Dictionary containing:
        - Overall metrics (loss, accuracy, w_mean, w_std, etc.)
        - Per-dataset metrics (if compute_per_dataset=True)
        - Dataset-type-specific metrics (ROC-AUC, cmAP for BirdSet; mAP for BEANS)
    """
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_multilabels = []
    all_logits = []
    all_w = []
    all_dataset_names = []

    for batch in dataloader:
        audio = batch["audio"].to(device)
        labels = batch["label"].to(device)
        multilabels = batch["multilabel"].to(device)
        sample_indices = batch["sample_indices"]
        metadata = batch["metadata"]
        dataset_names = batch.get("dataset_names", ["unknown"] * len(labels))

        prior_probs = prior_model.get_batch(sample_indices)
        prior_probs = torch.from_numpy(prior_probs).float().to(device)

        with autocast("cuda", dtype=torch.bfloat16):
            final_logits, audio_logits, w = model(audio, prior_probs, metadata, dataset_names)
            logits = final_logits if use_fusion else audio_logits
            loss = F.cross_entropy(logits, labels)

        total_loss += loss.item() * len(labels)
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_multilabels.append(multilabels.cpu().numpy())
        all_logits.append(logits.float().cpu())
        all_w.extend(w.float().cpu().numpy())
        all_dataset_names.extend(dataset_names)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_multilabels = np.concatenate(all_multilabels, axis=0)
    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_w = np.array(all_w)

    # Base metrics
    metrics = {
        "loss": total_loss / len(all_labels),
        "accuracy": accuracy_score(all_labels, all_preds) if HAS_SKLEARN else (all_preds == all_labels).mean(),
        "w_mean": float(all_w.mean()),
        "w_std": float(all_w.std()),
        "num_samples": len(all_labels),
    }

    # Determine dataset types present
    unique_datasets = set(all_dataset_names)
    has_beans = any(d.lower() in BEANS_DATASETS for d in unique_datasets)
    has_birdset = any(d.upper() in BIRDSET_DATASETS for d in unique_datasets)

    # Compute overall type-specific metrics
    if has_birdset and not has_beans:
        # Pure BirdSet evaluation
        birdset_metrics = compute_metrics_for_dataset_type(
            all_labels, all_multilabels, all_logits, all_preds, "birdset"
        )
        metrics.update(birdset_metrics)
    elif has_beans and not has_birdset:
        # Pure BEANS evaluation
        beans_metrics = compute_metrics_for_dataset_type(
            all_labels, all_multilabels, all_logits, all_preds, "beans"
        )
        metrics.update(beans_metrics)
    else:
        # Mixed - compute both for overall
        try:
            num_classes = all_logits.shape[1]
            probs = torch.softmax(torch.from_numpy(all_logits), dim=1).numpy()
            labels_onehot = np.eye(num_classes)[all_labels]
            metrics["mAP"] = average_precision_score(labels_onehot, probs, average="macro")
        except Exception:
            pass

    # Compute per-dataset metrics
    if compute_per_dataset and len(unique_datasets) > 1:
        per_dataset_metrics = {}
        for ds_name in unique_datasets:
            # Filter samples for this dataset
            mask = np.array([d == ds_name for d in all_dataset_names])
            if mask.sum() == 0:
                continue

            ds_labels = all_labels[mask]
            ds_multilabels = all_multilabels[mask]
            ds_logits = all_logits[mask]
            ds_preds = all_preds[mask]

            # Determine dataset type
            if ds_name.lower() in BEANS_DATASETS:
                ds_type = "beans"
            elif ds_name.upper() in BIRDSET_DATASETS:
                ds_type = "birdset"
            else:
                ds_type = "unknown"

            ds_metrics = compute_metrics_for_dataset_type(
                ds_labels, ds_multilabels, ds_logits, ds_preds, ds_type
            )
            ds_metrics["num_samples"] = int(mask.sum())
            per_dataset_metrics[ds_name] = ds_metrics

        metrics["per_dataset"] = per_dataset_metrics

        # Compute averaged metrics across datasets
        if per_dataset_metrics:
            avg_metrics = {}
            metric_keys = set()
            for ds_metrics in per_dataset_metrics.values():
                metric_keys.update(ds_metrics.keys())
            metric_keys.discard("num_samples")

            for key in metric_keys:
                values = [
                    ds_metrics[key]
                    for ds_metrics in per_dataset_metrics.values()
                    if key in ds_metrics
                ]
                if values:
                    avg_metrics[f"avg_{key}"] = np.mean(values)

            metrics.update(avg_metrics)

    return metrics


def print_evaluation_results(metrics: Dict[str, Any], title: str = "Evaluation Results"):
    """Pretty-print evaluation metrics."""
    print(f"\n{title}")
    print("-" * 50)

    # Print main metrics
    main_keys = ["loss", "accuracy", "mAP", "roc_auc", "cmAP", "top5_accuracy", "w_mean", "w_std"]
    for key in main_keys:
        if key in metrics:
            value = metrics[key]
            if key == "loss":
                print(f"  {key}: {value:.4f}")
            elif key in ["accuracy", "mAP", "roc_auc", "cmAP", "top5_accuracy"]:
                print(f"  {key}: {100*value:.2f}%")
            else:
                print(f"  {key}: {value:.4f}")

    # Print averaged metrics
    avg_keys = [k for k in metrics.keys() if k.startswith("avg_")]
    if avg_keys:
        print("\n  Averaged across datasets:")
        for key in avg_keys:
            value = metrics[key]
            metric_name = key[4:]  # Remove "avg_" prefix
            if metric_name in ["accuracy", "mAP", "roc_auc", "cmAP", "top5_accuracy"]:
                print(f"    {metric_name}: {100*value:.2f}%")
            else:
                print(f"    {metric_name}: {value:.4f}")

    # Print per-dataset metrics
    if "per_dataset" in metrics:
        print("\n  Per-dataset metrics:")
        for ds_name, ds_metrics in metrics["per_dataset"].items():
            print(f"\n    {ds_name} (n={ds_metrics.get('num_samples', '?')}):")
            for key, value in ds_metrics.items():
                if key == "num_samples":
                    continue
                if key in ["accuracy", "mAP", "roc_auc", "cmAP", "top5_accuracy"]:
                    print(f"      {key}: {100*value:.2f}%")
                else:
                    print(f"      {key}: {value:.4f}")


@torch.no_grad()
def evaluate_prior_only(
    prior_model: SpatioTemporalPriorModel,
    dataloader: DataLoader,
    device: str,
) -> Dict[str, Any]:
    """
    Evaluate prior-only model on dataset.

    Args:
        prior_model: SpatioTemporalPriorModel instance
        dataloader: Data loader for evaluation
        device: Device to evaluate on

    Returns:
        Dictionary containing metrics (loss, accuracy, etc.)
    """
    prior_model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_multilabels = []
    all_logits = []
    all_dataset_names = []

    for batch in dataloader:
        labels = batch["label"].to(device)
        multilabels = batch["multilabel"].to(device)
        metadata = batch["metadata"]
        dataset_names = batch.get("dataset_names", ["unknown"] * len(labels))

        probs = prior_model(metadata, dataset_names)
        logits = torch.log(probs + 1e-10)  # Convert to logits for consistent metric computation
        loss = F.cross_entropy(probs, labels)

        total_loss += loss.item() * len(labels)
        preds = probs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_multilabels.append(multilabels.cpu().numpy())
        all_logits.append(logits.float().cpu())
        all_dataset_names.extend(dataset_names)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_multilabels = np.concatenate(all_multilabels, axis=0)
    all_logits = torch.cat(all_logits, dim=0).numpy()

    # Base metrics
    metrics = {
        "loss": total_loss / len(all_labels),
        "accuracy": accuracy_score(all_labels, all_preds) if HAS_SKLEARN else (all_preds == all_labels).mean(),
        "num_samples": len(all_labels),
    }

    # Determine dataset types present
    unique_datasets = set(all_dataset_names)
    has_beans = any(d.lower() in BEANS_DATASETS for d in unique_datasets)
    has_birdset = any(d.upper() in BIRDSET_DATASETS for d in unique_datasets)

    # Compute type-specific metrics
    if has_birdset and not has_beans:
        birdset_metrics = compute_metrics_for_dataset_type(
            all_labels, all_multilabels, all_logits, all_preds, "birdset"
        )
        metrics.update(birdset_metrics)
    elif has_beans and not has_birdset:
        beans_metrics = compute_metrics_for_dataset_type(
            all_labels, all_multilabels, all_logits, all_preds, "beans"
        )
        metrics.update(beans_metrics)
    else:
        # Mixed - compute both
        try:
            num_classes = all_logits.shape[1]
            probs = torch.softmax(torch.from_numpy(all_logits), dim=1).numpy()
            labels_onehot = np.eye(num_classes)[all_labels]
            metrics["mAP"] = average_precision_score(labels_onehot, probs, average="macro")
        except Exception:
            pass

    return metrics


def print_model_comparison(
    audio_metrics: Dict[str, Any],
    prior_metrics: Dict[str, Any],
    finch_metrics: Dict[str, Any],
):
    """
    Print a side-by-side comparison table of the three model conditions.

    Args:
        audio_metrics: Metrics from audio-only model
        prior_metrics: Metrics from prior-only model
        finch_metrics: Metrics from FINCH (fused) model
    """
    print("\n")
    print("=" * 70)
    print("                     MODEL COMPARISON SUMMARY")
    print("=" * 70)

    # Define metrics to display and their display names
    metric_display = [
        ("accuracy", "Accuracy"),
        ("top5_accuracy", "Top-5 Accuracy"),
        ("mAP", "mAP"),
        ("roc_auc", "ROC-AUC"),
        ("cmAP", "cmAP"),
        ("loss", "Loss"),
    ]

    # Table header
    print("┌" + "─" * 20 + "┬" + "─" * 14 + "┬" + "─" * 14 + "┬" + "─" * 17 + "┐")
    print(f"│ {'Metric':<18} │ {'Audio-Only':^12} │ {'Prior-Only':^12} │ {'FINCH (Fused)':^15} │")
    print("├" + "─" * 20 + "┼" + "─" * 14 + "┼" + "─" * 14 + "┼" + "─" * 17 + "┤")

    # Table rows
    for metric_key, metric_name in metric_display:
        audio_val = audio_metrics.get(metric_key)
        prior_val = prior_metrics.get(metric_key)
        finch_val = finch_metrics.get(metric_key)

        # Skip if none of the models have this metric
        if audio_val is None and prior_val is None and finch_val is None:
            continue

        # Format values
        def format_val(val, is_loss=False):
            if val is None:
                return "-"
            if is_loss:
                return f"{val:.4f}"
            return f"{100 * val:.2f}%"

        is_loss = metric_key == "loss"
        audio_str = format_val(audio_val, is_loss)
        prior_str = format_val(prior_val, is_loss)
        finch_str = format_val(finch_val, is_loss)

        # Highlight best value (highest for metrics, lowest for loss)
        if audio_val is not None and prior_val is not None and finch_val is not None:
            vals = [audio_val, prior_val, finch_val]
            if is_loss:
                best_idx = vals.index(min(vals))
            else:
                best_idx = vals.index(max(vals))

            strs = [audio_str, prior_str, finch_str]
            strs[best_idx] = f"*{strs[best_idx][:-1]}*" if not is_loss else f"*{strs[best_idx]}*"
            audio_str, prior_str, finch_str = strs

        print(f"│ {metric_name:<18} │ {audio_str:^12} │ {prior_str:^12} │ {finch_str:^15} │")

    print("└" + "─" * 20 + "┴" + "─" * 14 + "┴" + "─" * 14 + "┴" + "─" * 17 + "┘")
    print("  * indicates best value for each metric")
    print()


# ============================================================================
# Dataset Loading
# ============================================================================


def load_dataset(
    dataset_name: str,
    data_root: Path,
    split: str = "train",
    **kwargs,
):
    """Load a dataset by name."""
    dataset_name = dataset_name.lower()

    if dataset_name == "cbi":
        data_dir = data_root / "beans" / "cbi"
        return CBIDataset(str(data_dir), split=split, **kwargs)
    elif dataset_name.upper() in BIRDSET_DATASETS:
        data_dir = data_root / "birdset"
        # Birdset only has test_5s audio available
        birdset_split = "test_5s" if split in ("train", "val") else split
        return BirdsetDataset(
            str(data_dir),
            dataset_name=dataset_name.upper(),
            split=birdset_split,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def create_dataloaders(
    datasets: List[str],
    data_root: Path,
    batch_size: int,
    val_split: float,
    num_workers: int,
    seed: int,
    ssw_max_samples: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, Dict[str, int], List[str]]:
    """
    Create train and validation dataloaders.

    Args:
        datasets: List of dataset names to load
        data_root: Root directory for data
        batch_size: Batch size for dataloaders
        val_split: Fraction of data to use for validation
        num_workers: Number of dataloader workers
        seed: Random seed for reproducibility
        ssw_max_samples: Max samples for SSW dataset (default: 20000)

    Returns:
        train_loader, val_loader, label_to_idx, species_list
    """
    all_datasets = []
    all_labels = set()

    # Load datasets
    for ds_name in datasets:
        try:
            # Pass subset parameters for Birdset datasets
            kwargs = {"subset_seed": seed}
            # Pass max_samples for SSW: None = default 20k, -1 = all, positive = use that value
            if ds_name.upper() == "SSW":
                kwargs["max_samples"] = ssw_max_samples  # Can be None, -1, or positive int

            ds = load_dataset(ds_name, data_root, **kwargs)
            all_datasets.append(ds)
            all_labels.update(ds.get_all_labels())
            print(f"Loaded {ds_name}: {len(ds)} samples")
        except Exception as e:
            print(f"Warning: Could not load {ds_name}: {e}")

    if not all_datasets:
        raise ValueError("No datasets could be loaded")

    # Build unified label mapping
    species_list = sorted(all_labels)
    label_to_idx = {s: i for i, s in enumerate(species_list)}
    print(f"Total classes: {len(species_list)}")

    # Apply label mapping to all datasets
    for ds in all_datasets:
        ds.label_to_idx = label_to_idx

    # Combine datasets
    if len(all_datasets) == 1:
        combined = all_datasets[0]
    else:
        combined = ConcatDataset(all_datasets)

    # Split into train/val
    total_size = len(combined)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        combined, [train_size, val_size], generator=generator
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

    return train_loader, val_loader, label_to_idx, species_list


# ============================================================================
# Checkpointing
# ============================================================================


def save_checkpoint(
    path: Path,
    model: FusionModel,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    phase: int,
    best_metric: float,
    config: Dict,
):
    """Save training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "phase": phase,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_metric": best_metric,
        "rng_state": get_rng_state(),
        "config": config,
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(path: Path, model: FusionModel, device: str) -> Dict:
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    set_rng_state(checkpoint["rng_state"])
    print(f"Checkpoint loaded: {path}")
    return checkpoint


# ============================================================================
# Main
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bio-acoustic classification training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset
    parser.add_argument(
        "--dataset", type=str, default="cbi",
        help="Dataset(s) to use. Options: cbi, PER, NES, UHH, HSN, SSW, SNE, all. "
             "Comma-separated for multiple."
    )
    parser.add_argument("--data_root", type=str, default=None, help="Data root directory")
    parser.add_argument("--priors_cache", type=str, default=None, help="Path to priors cache")
    parser.add_argument("--ssw_max_samples", type=int, default=20000,
                        help="Max samples for SSW dataset (default: 20000). Set to -1 to use all samples.")

    # Model
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "max", "cls"])
    parser.add_argument("--w_max", type=float, default=2.0)
    parser.add_argument("--init_w", type=float, default=0.0308)
    parser.add_argument("--init_temperature", type=float, default=0.5101)
    parser.add_argument("--init_epsilon", type=float, default=-0.049955)
    parser.add_argument("--gate_hidden_dim", type=int, default=64)

    # Training
    parser.add_argument("--phase1_epochs", type=int, default=10)
    parser.add_argument("--phase2_epochs", type=int, default=10)
    parser.add_argument("--phase1_lr", type=float, default=1e-3)
    parser.add_argument("--phase2_lr", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--variance_lambda", type=float, default=0.01)

    # Encoder fine-tuning
    parser.add_argument("--finetune_encoder", action="store_true", help="Fine-tune the audio encoder during Phase 1")
    parser.add_argument("--encoder_lr_factor", type=float, default=0.1, help="LR multiplier for encoder (default: 0.1 = 10x lower than classifier)")

    # Hardware
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    # Logging
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--log_interval", type=int, default=10)

    # Checkpointing
    parser.add_argument("--exp_dir", type=str, default=None, help="Experiment directory")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")

    # Spatio-temporal prior
    parser.add_argument("--train_prior", action="store_true", default=True, help="Train spatio-temporal prior model")
    parser.add_argument("--no_train_prior", action="store_false", dest="train_prior", help="Disable prior model training")
    parser.add_argument("--prior_epochs", type=int, default=20, help="Epochs for prior model training")
    parser.add_argument("--prior_lr", type=float, default=1e-3, help="Learning rate for prior model")
    parser.add_argument("--prior_hidden_dim", type=int, default=128, help="Hidden dim for prior model")

    # Evaluation
    parser.add_argument("--per_dataset_metrics", action="store_true", help="Compute per-dataset metrics")

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
        args.exp_dir = PROJECT_ROOT / "experiments" / timestamp
    else:
        args.exp_dir = Path(args.exp_dir)

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    (args.exp_dir / "checkpoints").mkdir(exist_ok=True)
    (args.exp_dir / "logs").mkdir(exist_ok=True)
    (args.exp_dir / "results").mkdir(exist_ok=True)

    # Save config
    config = vars(args).copy()
    config["data_root"] = str(config["data_root"])
    config["exp_dir"] = str(config["exp_dir"])
    with open(args.exp_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Device
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Parse datasets
    if args.dataset.lower() == "all":
        datasets = ["cbi"] + BIRDSET_DATASETS
    else:
        datasets = [d.strip() for d in args.dataset.split(",")]

    print(f"Datasets: {datasets}")

    # Create dataloaders
    # ssw_max_samples: None = use default (20k), -1 = use all samples, positive = use that value
    train_loader, val_loader, label_to_idx, species_list = create_dataloaders(
        datasets=datasets,
        data_root=args.data_root,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        seed=args.seed,
        ssw_max_samples=args.ssw_max_samples,
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

    # Initialize prior model
    cache_path = None
    if args.priors_cache:
        cache_path = Path(args.priors_cache)
    elif "cbi" in [d.lower() for d in datasets]:
        # Try to find CBI priors cache
        cbi_cache = CBIDataset.get_priors_cache_path(args.data_root / "beans" / "cbi")
        if cbi_cache:
            cache_path = cbi_cache

    prior_model = PriorWrapper(species_list, cache_path)

    # Initialize wandb
    use_wandb = HAS_WANDB and args.wandb_project is not None
    if use_wandb:
        run_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(project=args.wandb_project, name=run_name, config=config)

    # Phase 1: Train linear probe (optionally with encoder fine-tuning)
    probe_acc = train_phase1(
        model, train_loader, val_loader, prior_model, device,
        epochs=args.phase1_epochs,
        lr=args.phase1_lr,
        log_interval=args.log_interval,
        use_wandb=use_wandb,
        finetune_encoder=args.finetune_encoder,
        encoder_lr_factor=args.encoder_lr_factor,
    )

    # Freeze encoder after Phase 1 (if it was unfrozen)
    if args.finetune_encoder:
        model.encoder.set_trainable(False)
        print("Encoder frozen for Phase 2")

    # Save phase 1 checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "probe_accuracy": probe_acc,
        "label_to_idx": label_to_idx,
        "species_list": species_list,
    }, args.exp_dir / "checkpoints" / "phase1_best.pth")

    # Phase 2: Train gating network
    final_acc = train_phase2(
        model, train_loader, val_loader, prior_model, device,
        epochs=args.phase2_epochs,
        lr=args.phase2_lr,
        variance_lambda=args.variance_lambda,
        log_interval=args.log_interval,
        use_wandb=use_wandb,
    )

    # Train spatio-temporal prior model (after the two-stage pipeline)
    st_prior_model = None
    st_prior_acc = None
    prior_only_metrics = None
    if args.train_prior:
        st_prior_model = SpatioTemporalPriorModel(
            num_classes=num_classes,
            hidden_dim=args.prior_hidden_dim,
        ).to(device)

        st_prior_acc = train_spatio_temporal_prior(
            st_prior_model,
            train_loader,
            val_loader,
            device,
            epochs=args.prior_epochs,
            lr=args.prior_lr,
            log_interval=args.log_interval,
            use_wandb=use_wandb,
        )

        # Evaluate prior-only model
        prior_only_metrics = evaluate_prior_only(
            st_prior_model,
            val_loader,
            device,
        )

        # Save prior model
        torch.save({
            "model_state_dict": st_prior_model.state_dict(),
            "accuracy": st_prior_acc,
            "num_classes": num_classes,
            "hidden_dim": args.prior_hidden_dim,
        }, args.exp_dir / "checkpoints" / "st_prior_model.pth")

    # Final evaluation with per-dataset metrics
    final_metrics = evaluate(
        model, val_loader, prior_model, device,
        use_fusion=True,
        compute_per_dataset=args.per_dataset_metrics,
    )

    # Also evaluate audio-only for comparison
    audio_only_metrics = evaluate(
        model, val_loader, prior_model, device,
        use_fusion=False,
        compute_per_dataset=args.per_dataset_metrics,
    )

    # Save final model
    save_dict = {
        "model_state_dict": model.state_dict(),
        "probe_accuracy": probe_acc,
        "final_accuracy": final_acc,
        "final_temperature": model.temperature.item(),
        "final_epsilon": model.epsilon.item(),
        "final_w_mean": final_metrics['w_mean'],
        "final_w_std": final_metrics['w_std'],
        "label_to_idx": label_to_idx,
        "species_list": species_list,
    }
    if st_prior_acc is not None:
        save_dict["st_prior_accuracy"] = st_prior_acc
    torch.save(save_dict, args.exp_dir / "checkpoints" / "final_model.pth")

    # Prepare results for JSON (filter out non-serializable items)
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return obj

    # Save metrics
    results = {
        "probe_accuracy": probe_acc,
        "final_accuracy": final_acc,
        "st_prior_accuracy": st_prior_acc,
        "final_metrics": make_serializable(final_metrics),
        "audio_only_metrics": make_serializable(audio_only_metrics),
        "prior_only_metrics": make_serializable(prior_only_metrics) if prior_only_metrics else None,
        "config": config,
    }
    with open(args.exp_dir / "results" / "metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(f"\nPhase 1 (Linear Probe) Accuracy: {100*probe_acc:.2f}%")
    print(f"Phase 2 (Fused Model)  Accuracy: {100*final_acc:.2f}%")
    if st_prior_acc is not None:
        print(f"Prior Model Accuracy:            {100*st_prior_acc:.2f}%")

    print(f"\nFinal Parameters:")
    print(f"  temperature: {model.temperature.item():.4f}")
    print(f"  epsilon: {model.epsilon.item():.6f}")
    print(f"  w_mean: {final_metrics['w_mean']:.4f}")
    print(f"  w_std: {final_metrics['w_std']:.4f}")

    # Print comparison table if prior model was trained
    if prior_only_metrics is not None:
        print_model_comparison(
            audio_metrics=audio_only_metrics,
            prior_metrics=prior_only_metrics,
            finch_metrics=final_metrics,
        )
    else:
        # Print detailed metrics (legacy output when prior not trained)
        print_evaluation_results(final_metrics, "Final Fused Model Metrics")
        print_evaluation_results(audio_only_metrics, "Audio-Only Model Metrics (for comparison)")

    print(f"\nResults saved to: {args.exp_dir}")

    if use_wandb:
        # Log final metrics
        log_dict = {
            "final/probe_accuracy": probe_acc,
            "final/fused_accuracy": final_acc,
        }
        if st_prior_acc is not None:
            log_dict["final/st_prior_accuracy"] = st_prior_acc

        # Log main final metrics for each model type
        for k, v in final_metrics.items():
            if k not in ["per_dataset"] and not isinstance(v, dict):
                log_dict[f"final/finch/{k}"] = v

        for k, v in audio_only_metrics.items():
            if k not in ["per_dataset"] and not isinstance(v, dict):
                log_dict[f"final/audio_only/{k}"] = v

        if prior_only_metrics is not None:
            for k, v in prior_only_metrics.items():
                if k not in ["per_dataset"] and not isinstance(v, dict):
                    log_dict[f"final/prior_only/{k}"] = v

        # Log per-dataset metrics
        if "per_dataset" in final_metrics:
            for ds_name, ds_metrics in final_metrics["per_dataset"].items():
                for k, v in ds_metrics.items():
                    log_dict[f"final/{ds_name}/{k}"] = v

        wandb.log(log_dict)
        wandb.finish()


if __name__ == "__main__":
    main()
