"""
Stage D Training: Two-phase ProtoPNet + gating network learning.

Phase 1: Train ProtoPNet head (AudioProtoPNet-style classifier) with frozen encoder
Phase 2: Freeze ProtoPNet head, train gating network for w(a,x,t)

Model: final_logits = audio_logits / T + w(a,x,t) * log(prior + eps)

The ProtoPNet head learns J prototypes per class (default: 10) in the encoder's
feature space. Classification is based on cosine similarity to these prototypes,
with global max-pooling per class. This provides interpretability by showing
which learned prototypical patterns a sample most resembles.

Based on: "AudioProtoPNet: An interpretable deep learning model for bird sound
classification" (Heinrich et al., 2024) - arXiv:2404.10420

Key features:
- AudioProtoPNet-style prototype learning for species classification
- Uses gating network instead of scalar w for prior fusion
- Variance regularization to prevent weight collapse
- Logs gating weight statistics during training
"""

import argparse
import math
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
import resampy

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, top_k_accuracy_score

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "NatureLM-audio"))

# Suppress audio warnings
warnings.filterwarnings("ignore", category=UserWarning)

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


# ============================================================================
# Dataset
# ============================================================================

class BirdAudioDataset(Dataset):
    """Bird audio dataset with location/time metadata."""

    def __init__(
        self,
        audio_dir: Path,
        metadata_df: pd.DataFrame,
        label_to_idx: dict,
        max_length_seconds: float = 10.0,
        sample_rate: int = 16000,
    ):
        self.audio_dir = Path(audio_dir)
        self.metadata_df = metadata_df.copy()
        self.original_indices = self.metadata_df.index.values
        self.metadata_df = self.metadata_df.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.sample_rate = sample_rate
        self.max_samples = int(max_length_seconds * sample_rate)
        self.cache = {}

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]

        row = self.metadata_df.iloc[idx]
        audio_path = self.audio_dir / row["ebird_code"] / row["filename"]
        audio_path = str(audio_path).replace(".mp3", ".ogg")

        try:
            audio, sr = sf.read(audio_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load: {audio_path}") from e

        # Convert to mono
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Resample if needed
        if sr != self.sample_rate:
            audio = resampy.resample(audio, sr, self.sample_rate)

        # Pad or truncate
        if len(audio) > self.max_samples:
            audio = audio[:self.max_samples]
        elif len(audio) < self.max_samples:
            audio = np.pad(audio, (0, self.max_samples - len(audio)))

        # Parse metadata
        date_str = row.get("date", "")
        try:
            date = pd.to_datetime(date_str)
            day_of_year = date.timetuple().tm_yday
        except:
            day_of_year = 182

        # Parse hour
        hour = row.get("time", "12:00")
        try:
            if isinstance(hour, str) and ":" in hour:
                hour = int(hour.split(":")[0])
            else:
                hour = 12
        except:
            hour = 12

        sample = {
            "audio": torch.from_numpy(audio.astype(np.float32)),
            "label": self.label_to_idx[row["species"]],
            "metadata": {
                "latitude": float(row.get("latitude", 0.0) or 0.0),
                "longitude": float(row.get("longitude", 0.0) or 0.0),
                "day_of_year": day_of_year,
                "hour": hour,
                "date": date_str,
            },
            "sample_idx": self.original_indices[idx],
        }

        self.cache[idx] = sample
        return sample


def collate_fn(batch):
    """Collate batch with metadata."""
    return {
        "audio": torch.stack([b["audio"] for b in batch]),
        "label": torch.tensor([b["label"] for b in batch]),
        "metadata": [b["metadata"] for b in batch],
        "sample_indices": [b["sample_idx"] for b in batch],
    }


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

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            audio_embeds, audio_atts = self.naturelm.encode_audio(audio)

        # Pool over sequence dimension
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

    Input: 12-dim vector
        - 3 audio features: max_prob, entropy, margin
        - 3 prior features: max_prob, entropy, margin
        - 6 metadata features: sin/cos(day), sin/cos(hour), lat, lon

    Output: scalar w in [0, w_max]
    """

    def __init__(
        self,
        input_dim: int = 12,
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

        # Initialize final layer bias so sigmoid outputs ~init_w/w_max
        # sigmoid(bias) * w_max = init_w => bias = logit(init_w / w_max)
        target_sigmoid = init_w / w_max
        target_sigmoid = max(1e-6, min(1 - 1e-6, target_sigmoid))  # Clamp to valid range
        init_bias = math.log(target_sigmoid / (1 - target_sigmoid))  # Inverse sigmoid

        # Get final layer and set bias
        final_layer = self.mlp[-1]
        nn.init.zeros_(final_layer.weight)
        nn.init.constant_(final_layer.bias, init_bias)
        print(f"Gating network initialized: target w={init_w:.4f}, bias={init_bias:.4f}")

    def forward(
        self,
        audio_features: torch.Tensor,
        prior_features: torch.Tensor,
        metadata: list,
    ) -> torch.Tensor:
        """
        Args:
            audio_features: (batch_size, 3) - [max_prob, entropy, margin]
            prior_features: (batch_size, 3) - [max_prob, entropy, margin]
            metadata: list of dicts with 'day_of_year', 'hour', 'latitude', 'longitude'

        Returns:
            w: (batch_size,) - gate weights in [0, w_max]
        """
        device = audio_features.device

        # Extract metadata safely
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

        # Normalize lat/lon to [-1, 1]
        lat_norm = lat / 90.0
        lon_norm = lon / 180.0

        # Sin/cos encoding for cyclical features
        day_sin = torch.sin(2 * math.pi * day_of_year / 365.25)
        day_cos = torch.cos(2 * math.pi * day_of_year / 365.25)
        hour_sin = torch.sin(2 * math.pi * hour / 24.0)
        hour_cos = torch.cos(2 * math.pi * hour / 24.0)

        # Concatenate all features: [audio(3), prior(3), metadata(6)] = 12
        meta_features = torch.stack(
            [day_sin, day_cos, hour_sin, hour_cos, lat_norm, lon_norm], dim=1
        )
        combined = torch.cat([audio_features, prior_features, meta_features], dim=1)

        # MLP to gate logit
        gate_logit = self.mlp(combined).squeeze(-1)

        # Sigmoid and scale to [0, w_max]
        w = torch.sigmoid(gate_logit) * self.w_max

        return w


class ProtoPNetHead(nn.Module):
    """
    AudioProtoPNet-style classification head using learned prototypes.

    Based on: "AudioProtoPNet: An interpretable deep learning model for
    bird sound classification" (Heinrich et al., 2024)

    Key features:
    - Learns J prototypes per class in the feature space
    - Computes cosine similarity between input features and all prototypes
    - Global max-pooling over each class's prototypes to get class presence scores
    - Optional learnable final layer for combining prototype similarities

    Args:
        input_dim: Dimension of input features from encoder
        num_classes: Number of output classes
        num_prototypes_per_class: Number of prototypes to learn per class (default: 10)
        use_final_layer: If True, use learnable weights in final layer;
                         if False, use identity (1s for own class, 0s for others)
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_prototypes_per_class: int = 10,
        use_final_layer: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_prototypes_per_class = num_prototypes_per_class
        self.num_prototypes = num_classes * num_prototypes_per_class
        self.use_final_layer = use_final_layer

        # Learnable prototypes: (num_prototypes, input_dim)
        # Each prototype is a point in the embedding space
        self.prototypes = nn.Parameter(
            torch.randn(self.num_prototypes, input_dim) * 0.1
        )

        # Prototype-to-class mapping (fixed): which prototype belongs to which class
        # prototype_class_identity[i, c] = 1 if prototype i belongs to class c
        prototype_class_identity = torch.zeros(self.num_prototypes, num_classes)
        for c in range(num_classes):
            start_idx = c * num_prototypes_per_class
            end_idx = start_idx + num_prototypes_per_class
            prototype_class_identity[start_idx:end_idx, c] = 1.0
        self.register_buffer("prototype_class_identity", prototype_class_identity)

        # Final layer: maps prototype similarities to class logits
        # In original ProtoPNet, this ensures predictions for a class are based
        # only on the prototypes of that class
        if use_final_layer:
            # Learnable weights, initialized to class identity
            self.final_layer = nn.Linear(self.num_prototypes, num_classes, bias=False)
            # Initialize so that each class only looks at its own prototypes initially
            with torch.no_grad():
                self.final_layer.weight.copy_(prototype_class_identity.T)
        else:
            self.final_layer = None

        print(f"ProtoPNetHead initialized:")
        print(f"  num_classes: {num_classes}")
        print(f"  prototypes_per_class: {num_prototypes_per_class}")
        print(f"  total_prototypes: {self.num_prototypes}")
        print(f"  prototype_dim: {input_dim}")
        print(f"  use_final_layer: {use_final_layer}")

    def _cosine_similarity(
        self,
        features: torch.Tensor,
        prototypes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cosine similarity between features and prototypes.

        Args:
            features: (batch_size, input_dim)
            prototypes: (num_prototypes, input_dim)

        Returns:
            similarities: (batch_size, num_prototypes)
        """
        # Normalize to unit length (as per AudioProtoPNet)
        features_norm = F.normalize(features, p=2, dim=1)  # (B, D)
        prototypes_norm = F.normalize(prototypes, p=2, dim=1)  # (P, D)

        # Cosine similarity: dot product of normalized vectors
        similarities = torch.mm(features_norm, prototypes_norm.T)  # (B, P)

        return similarities

    def forward(self, features: torch.Tensor) -> tuple:
        """
        Forward pass through the ProtoPNet head.

        Args:
            features: (batch_size, input_dim) - encoded audio features

        Returns:
            logits: (batch_size, num_classes) - classification logits
            prototype_similarities: (batch_size, num_prototypes) - similarity to each prototype
            class_prototype_scores: (batch_size, num_classes) - max similarity per class
        """
        batch_size = features.size(0)

        # Compute similarity to all prototypes
        # Shape: (batch_size, num_prototypes)
        prototype_similarities = self._cosine_similarity(features, self.prototypes)

        # Global max-pooling per class: take max similarity among each class's prototypes
        # This captures "how much does this sample look like any prototype of class c?"
        # Shape: (batch_size, num_classes)
        class_prototype_scores = torch.zeros(
            batch_size, self.num_classes, device=features.device, dtype=features.dtype
        )
        for c in range(self.num_classes):
            start_idx = c * self.num_prototypes_per_class
            end_idx = start_idx + self.num_prototypes_per_class
            class_scores, _ = prototype_similarities[:, start_idx:end_idx].max(dim=1)
            class_prototype_scores[:, c] = class_scores

        # Compute final logits
        if self.final_layer is not None:
            # Learnable combination of all prototype similarities
            logits = self.final_layer(prototype_similarities)
        else:
            # Use max-pooled class scores directly as logits
            logits = class_prototype_scores

        return logits, prototype_similarities, class_prototype_scores

    def get_prototype_class(self, prototype_idx: int) -> int:
        """Get which class a prototype belongs to."""
        return prototype_idx // self.num_prototypes_per_class

    def get_class_prototypes(self, class_idx: int) -> torch.Tensor:
        """Get all prototypes for a specific class."""
        start_idx = class_idx * self.num_prototypes_per_class
        end_idx = start_idx + self.num_prototypes_per_class
        return self.prototypes[start_idx:end_idx]


class StageDModel(nn.Module):
    """
    Fusion model with gating network for Stage D training.

    Uses ProtoPNet head (AudioProtoPNet-style) for species classification.

    final_logits = audio_logits / temperature + w(a,x,t) * log(prior + epsilon)
    """

    def __init__(
        self,
        encoder: FrozenAudioEncoder,
        num_classes: int,
        w_max: float = 2.0,
        init_w: float = 0.0308,
        init_temperature: float = 0.5101,
        init_epsilon: float = -0.049955,
        gate_hidden_dim: int = 64,
        num_prototypes_per_class: int = 10,
        use_protopnet_final_layer: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes

        # ProtoPNet classification head (replaces linear probe)
        self.classifier = ProtoPNetHead(
            input_dim=encoder.encoder_dim,
            num_classes=num_classes,
            num_prototypes_per_class=num_prototypes_per_class,
            use_final_layer=use_protopnet_final_layer,
        )

        # Learnable fusion parameters
        self.temperature = nn.Parameter(torch.tensor(init_temperature))
        self.epsilon = nn.Parameter(torch.tensor(init_epsilon))

        # Gating network for w(a,x,t)
        self.gate_network = PriorGatingNetwork(
            input_dim=12,
            hidden_dim=gate_hidden_dim,
            w_max=w_max,
            init_w=init_w,
        )

        print(f"StageDModel initialized:")
        print(f"  temperature: {init_temperature:.4f}")
        print(f"  epsilon: {init_epsilon:.6f}")
        print(f"  w_max: {w_max:.2f}")
        print(f"  init_w: {init_w:.4f}")
        print(f"  classifier: ProtoPNet ({num_prototypes_per_class} prototypes/class)")

    def compute_audio_features(self, audio_logits: torch.Tensor) -> torch.Tensor:
        """Extract features from audio logits for gating."""
        probs = F.softmax(audio_logits, dim=1)
        max_prob = probs.max(dim=1)[0]
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)

        # Margin: difference between top-1 and top-2
        top2_probs = torch.topk(probs, k=2, dim=1)[0]
        margin = top2_probs[:, 0] - top2_probs[:, 1]

        return torch.stack([max_prob, entropy, margin], dim=1)

    def compute_prior_features(self, prior_probs: torch.Tensor) -> torch.Tensor:
        """Extract features from prior probabilities for gating."""
        max_prob = prior_probs.max(dim=1)[0]
        entropy = -(prior_probs * torch.log(prior_probs + 1e-10)).sum(dim=1)

        # Margin: difference between top-1 and top-2
        top2_probs = torch.topk(prior_probs, k=2, dim=1)[0]
        margin = top2_probs[:, 0] - top2_probs[:, 1]

        return torch.stack([max_prob, entropy, margin], dim=1)

    def forward(
        self,
        audio: torch.Tensor,
        prior_probs: torch.Tensor,
        metadata: list = None,
        return_prototypes: bool = False,
    ):
        """
        Args:
            audio: (batch_size, audio_length)
            prior_probs: (batch_size, num_classes)
            metadata: list of metadata dicts for gating
            return_prototypes: if True, also return prototype similarities

        Returns:
            final_logits: (batch_size, num_classes)
            audio_logits: (batch_size, num_classes)
            w: (batch_size,) - gating weights for logging
            (optional) prototype_info: dict with prototype similarities if return_prototypes=True
        """
        # Encode audio and classify using ProtoPNet head
        features = self.encoder(audio)
        audio_logits, prototype_sims, class_proto_scores = self.classifier(features.float())

        # Temperature-scaled audio logits
        temp = self.temperature.abs().clamp(min=1e-8)
        audio_logits_scaled = audio_logits / temp

        # Compute gating weight w(a,x,t)
        if metadata is not None:
            audio_features = self.compute_audio_features(audio_logits_scaled)
            prior_features = self.compute_prior_features(prior_probs)
            w = self.gate_network(audio_features, prior_features, metadata)
        else:
            # Fallback to mean initial weight if no metadata
            w = torch.full(
                (audio_logits.size(0),),
                self.gate_network.init_w,
                device=audio_logits.device,
            )

        # Safe epsilon: ensure prior + eps > 0 to avoid log(0)
        # Clamp epsilon to ensure prior_probs + eps is always positive
        min_prior = prior_probs.min()
        safe_eps = self.epsilon.clamp(min=-min_prior.item() + 1e-8)
        log_prior = torch.log(prior_probs + safe_eps)

        # Fused logits
        final_logits = audio_logits_scaled + w.unsqueeze(1) * log_prior

        if return_prototypes:
            prototype_info = {
                "prototype_similarities": prototype_sims,
                "class_prototype_scores": class_proto_scores,
            }
            return final_logits, audio_logits, w, prototype_info

        return final_logits, audio_logits, w


# ============================================================================
# Prior Model
# ============================================================================

class PriorWrapper:
    """Wrapper for eBird prior with pre-computed cache support."""

    def __init__(self, species_list: list, cache_path: Path = None, df_index=None):
        self.species_list = species_list
        self.num_species = len(species_list)

        if cache_path and Path(cache_path).exists():
            print(f"Loading prior cache from {cache_path}...")
            cache_path = Path(cache_path)

            if cache_path.suffix == '.h5':
                if not HAS_H5PY:
                    raise ImportError("h5py required to load .h5 cache files")
                with h5py.File(cache_path, 'r') as f:
                    self.prior_matrix = f['priors'][:]
                    species_raw = f['species_list'][:]
                    self.cached_species = [s.decode('utf-8') if isinstance(s, bytes) else s
                                           for s in species_raw]
                    self.cached_indices = f['sample_indices'][:]
            else:
                npz = np.load(cache_path, allow_pickle=True)
                self.prior_matrix = npz['priors']
                self.cached_species = [s.decode('utf-8') if isinstance(s, bytes) else s
                                       for s in npz['species_list']]
                self.cached_indices = npz['sample_indices']

            self.index_map = {idx: i for i, idx in enumerate(self.cached_indices)}
            self.use_cache = True
            print(f"Loaded cache: {len(self.cached_indices)} samples")

            # Reindex prior_matrix if cached species don't match current species
            if self.cached_species != self.species_list:
                print(f"Reindexing priors: cache has {len(self.cached_species)} species, "
                      f"current dataset has {len(self.species_list)} species")
                cached_species_to_idx = {s: i for i, s in enumerate(self.cached_species)}
                new_prior_matrix = np.zeros((self.prior_matrix.shape[0], self.num_species),
                                            dtype=self.prior_matrix.dtype)
                for i, species in enumerate(self.species_list):
                    if species in cached_species_to_idx:
                        new_prior_matrix[:, i] = self.prior_matrix[:, cached_species_to_idx[species]]
                    else:
                        print(f"Warning: species '{species}' not found in cache, using uniform prior")
                        new_prior_matrix[:, i] = 1.0 / self.num_species
                self.prior_matrix = new_prior_matrix
                # Renormalize rows to sum to 1
                row_sums = self.prior_matrix.sum(axis=1, keepdims=True)
                row_sums = np.where(row_sums > 0, row_sums, 1.0)
                self.prior_matrix = self.prior_matrix / row_sums
        else:
            self.use_cache = False
            print("No cache found, using uniform prior")

    def get_batch(self, sample_indices: list) -> np.ndarray:
        """Get prior probabilities for a batch."""
        batch_size = len(sample_indices)

        if not self.use_cache:
            # Uniform prior fallback
            return np.ones((batch_size, self.num_species), dtype=np.float32) / self.num_species

        priors = np.zeros((batch_size, self.num_species), dtype=np.float32)
        for i, idx in enumerate(sample_indices):
            if idx in self.index_map:
                priors[i] = self.prior_matrix[self.index_map[idx]]
            else:
                priors[i] = 1.0 / self.num_species  # Fallback to uniform

        return np.clip(priors, 1e-8, 1.0)


# ============================================================================
# Training Functions
# ============================================================================

def train_phase1(
    model: StageDModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    prior_model: PriorWrapper,
    device: str,
    epochs: int,
    lr: float,
    use_wandb: bool = False,
):
    """
    Phase 1: Train ProtoPNet head with frozen encoder.
    Only the ProtoPNet classifier (prototypes + final layer) is trained.
    """
    print("\n" + "=" * 60)
    print("PHASE 1: Training ProtoPNet Head")
    print("=" * 60)
    print(f"  Prototypes per class: {model.classifier.num_prototypes_per_class}")
    print(f"  Total prototypes: {model.classifier.num_prototypes}")

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
        # Training
        model.train()
        model.encoder.eval()  # Keep encoder in eval mode

        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Phase 1 Epoch {epoch}/{epochs}")
        for batch in pbar:
            audio = batch["audio"].to(device)
            labels = batch["label"].to(device)
            sample_indices = batch["sample_indices"]

            # Get prior (using cache)
            prior_probs = prior_model.get_batch(sample_indices)
            prior_probs = torch.from_numpy(prior_probs).float().to(device)

            optimizer.zero_grad()

            with autocast("cuda", dtype=torch.bfloat16):
                # In phase 1, we train on audio_logits only (no fusion)
                _, audio_logits, _ = model(audio, prior_probs, batch["metadata"])
                loss = F.cross_entropy(audio_logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * len(labels)
            preds = audio_logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

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

    # Restore best classifier weights
    if best_state:
        model.classifier.load_state_dict(best_state)

    print(f"\nPhase 1 complete. Best ProtoPNet head accuracy: {100*best_acc:.2f}%")
    return best_acc


def train_phase2(
    model: StageDModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    prior_model: PriorWrapper,
    device: str,
    epochs: int,
    lr: float,
    variance_lambda: float = 0.01,
    use_wandb: bool = False,
):
    """
    Phase 2: Freeze ProtoPNet head, train gating network and fusion parameters.

    Loss = CrossEntropy - variance_lambda * Var(w)

    The variance term encourages the gating weight to not collapse to a constant.
    """
    print("\n" + "=" * 60)
    print("PHASE 2: Training Gating Network")
    print("=" * 60)
    print(f"  Variance regularization lambda: {variance_lambda}")

    # Freeze classifier, train only fusion parameters (gating network + temp/eps)
    for param in model.parameters():
        param.requires_grad = False

    # Enable training for gating network
    for param in model.gate_network.parameters():
        param.requires_grad = True

    # Enable training for temperature and epsilon
    model.temperature.requires_grad = True
    model.epsilon.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable}")

    # Collect trainable parameters
    fusion_params = list(model.gate_network.parameters()) + [model.temperature, model.epsilon]
    optimizer = torch.optim.AdamW(fusion_params, lr=lr, weight_decay=0.001)
    scaler = GradScaler("cuda")

    best_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        model.encoder.eval()

        total_loss = 0.0
        total_ce_loss = 0.0
        total_var_loss = 0.0
        correct = 0
        total = 0

        # Track gating weight statistics
        all_w = []

        pbar = tqdm(train_loader, desc=f"Phase 2 Epoch {epoch}/{epochs}")
        for batch in pbar:
            audio = batch["audio"].to(device)
            labels = batch["label"].to(device)
            sample_indices = batch["sample_indices"]
            metadata = batch["metadata"]

            prior_probs = prior_model.get_batch(sample_indices)
            prior_probs = torch.from_numpy(prior_probs).float().to(device)

            optimizer.zero_grad()

            with autocast("cuda", dtype=torch.bfloat16):
                final_logits, _, w = model(audio, prior_probs, metadata)

                # Cross-entropy loss
                ce_loss = F.cross_entropy(final_logits, labels)

                # Variance loss: encourage non-zero variance in gating weight
                # We SUBTRACT variance to MAXIMIZE it (prevent collapse to constant)
                w_float = w.float()
                w_var = w_float.var()
                var_loss = -variance_lambda * w_var

                # Total loss
                loss = ce_loss + var_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Track statistics
            total_loss += loss.item() * len(labels)
            total_ce_loss += ce_loss.item() * len(labels)
            total_var_loss += var_loss.item() * len(labels)
            preds = final_logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

            # Collect w values for epoch statistics
            all_w.extend(w.detach().cpu().float().numpy().tolist())

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{100*correct/total:.2f}%",
                w_mean=f"{w.mean().item():.4f}",
                w_std=f"{w.std().item():.4f}"
            )

        train_loss = total_loss / total
        train_ce_loss = total_ce_loss / total
        train_var_loss = total_var_loss / total
        train_acc = correct / total

        # Compute epoch-level w statistics
        all_w = np.array(all_w)
        w_mean = all_w.mean()
        w_std = all_w.std()
        w_min = all_w.min()
        w_max = all_w.max()

        # Validation
        val_metrics = evaluate(model, val_loader, prior_model, device, use_fusion=True)

        print(f"  Train Loss: {train_loss:.4f} (CE: {train_ce_loss:.4f}, Var: {train_var_loss:.4f})")
        print(f"  Train Acc: {100*train_acc:.2f}%")
        print(f"  Val   Loss: {val_metrics['loss']:.4f}, Acc: {100*val_metrics['accuracy']:.2f}%")
        print(f"  Gating weight w: mean={w_mean:.4f}, std={w_std:.4f}, min={w_min:.4f}, max={w_max:.4f}")
        print(f"  Temperature={model.temperature.item():.4f}, Epsilon={model.epsilon.item():.6f}")

        if use_wandb and HAS_WANDB:
            wandb.log({
                "phase2/epoch": epoch,
                "phase2/train_loss": train_loss,
                "phase2/train_ce_loss": train_ce_loss,
                "phase2/train_var_loss": train_var_loss,
                "phase2/train_acc": train_acc,
                "phase2/val_loss": val_metrics['loss'],
                "phase2/val_acc": val_metrics['accuracy'],
                "phase2/val_top5_acc": val_metrics['top5_accuracy'],
                # Gating weight statistics
                "phase2/w_mean": w_mean,
                "phase2/w_std": w_std,
                "phase2/w_min": w_min,
                "phase2/w_max": w_max,
                "phase2/w_variance": w_std ** 2,
                # Model parameters
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

    # Restore best state
    if best_state:
        model.gate_network.load_state_dict(best_state["gate_network"])
        model.temperature.data = best_state["temperature"]
        model.epsilon.data = best_state["epsilon"]

    print(f"\nPhase 2 complete. Best fused accuracy: {100*best_acc:.2f}%")
    return best_acc


@torch.no_grad()
def evaluate(
    model: StageDModel,
    dataloader: DataLoader,
    prior_model: PriorWrapper,
    device: str,
    use_fusion: bool = True,
):
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

    return {
        "loss": total_loss / len(all_labels),
        "accuracy": accuracy_score(all_labels, all_preds),
        "top5_accuracy": top_k_accuracy_score(all_labels, all_logits, k=5),
        "w_mean": all_w.mean(),
        "w_std": all_w.std(),
        "w_min": all_w.min(),
        "w_max": all_w.max(),
    }


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Stage D: ProtoPNet + gating network training")

    # Data
    parser.add_argument("--data_dir", type=str, default="../Data/cbi")
    parser.add_argument("--priors_cache", type=str, default=None)

    # Model
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "max", "cls"])
    parser.add_argument("--w_max", type=float, default=2.0, help="Maximum gate weight")
    parser.add_argument("--init_w", type=float, default=0.0308, help="Initial gating weight")
    parser.add_argument("--init_temperature", type=float, default=0.5101, help="Initial temperature")
    parser.add_argument("--init_epsilon", type=float, default=-0.049955, help="Initial epsilon")
    parser.add_argument("--gate_hidden_dim", type=int, default=64, help="Gating network hidden dim")

    # ProtoPNet head parameters
    parser.add_argument("--num_prototypes_per_class", type=int, default=10,
                        help="Number of prototypes per class (AudioProtoPNet style)")
    parser.add_argument("--no_protopnet_final_layer", action="store_true",
                        help="If set, use max-pooled prototype scores directly instead of learned final layer")

    # Training
    parser.add_argument("--phase1_epochs", type=int, default=10)
    parser.add_argument("--phase2_epochs", type=int, default=10)
    parser.add_argument("--phase1_lr", type=float, default=1e-3)
    parser.add_argument("--phase2_lr", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--variance_lambda", type=float, default=0.01,
                        help="Coefficient for variance regularization (higher = more variance)")

    # Hardware
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)

    # Logging
    parser.add_argument("--wandb_project", type=str, default="stage-d-training")
    parser.add_argument("--no_wandb", action="store_true")

    # Checkpointing
    parser.add_argument("--save_dir", type=str, default="../checkpoints_stage_d")

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load data
    data_dir = Path(args.data_dir).resolve()
    train_csv = pd.read_csv(data_dir / "train.csv")

    # Filter to files that exist
    print("Checking for existing audio files...")
    audio_dir = data_dir / "train_audio"
    valid_mask = []
    for _, row in tqdm(train_csv.iterrows(), total=len(train_csv), desc="Validating files"):
        audio_path = audio_dir / row["ebird_code"] / row["filename"]
        audio_path = Path(str(audio_path).replace(".mp3", ".ogg"))
        valid_mask.append(audio_path.is_file())

    train_csv = train_csv[valid_mask]
    print(f"Valid samples: {len(train_csv)}")

    # Create label mapping
    all_species = sorted(train_csv["species"].unique())
    label_to_idx = {s: i for i, s in enumerate(all_species)}
    idx_to_label = {i: s for s, i in label_to_idx.items()}
    num_classes = len(all_species)
    print(f"Number of classes: {num_classes}")

    # Split data
    train_split, val_split = train_test_split(
        train_csv, test_size=args.val_split,
        stratify=train_csv["species"], random_state=args.seed
    )

    # Create datasets
    train_dataset = BirdAudioDataset(audio_dir, train_split, label_to_idx)
    val_dataset = BirdAudioDataset(audio_dir, val_split, label_to_idx)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn
    )

    # Initialize model with ProtoPNet head
    encoder = FrozenAudioEncoder(pooling=args.pooling)
    model = StageDModel(
        encoder=encoder,
        num_classes=num_classes,
        w_max=args.w_max,
        init_w=args.init_w,
        init_temperature=args.init_temperature,
        init_epsilon=args.init_epsilon,
        gate_hidden_dim=args.gate_hidden_dim,
        num_prototypes_per_class=args.num_prototypes_per_class,
        use_protopnet_final_layer=not args.no_protopnet_final_layer,
    ).to(device)

    # Initialize prior model
    cache_path = Path(args.priors_cache).resolve() if args.priors_cache else None
    prior_model = PriorWrapper(list(idx_to_label.values()), cache_path)

    # Initialize wandb
    use_wandb = HAS_WANDB and not args.no_wandb
    if use_wandb:
        run_name = f"stage_d_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Train ProtoPNet head
    protopnet_acc = train_phase1(
        model, train_loader, val_loader, prior_model, device,
        epochs=args.phase1_epochs, lr=args.phase1_lr, use_wandb=use_wandb
    )

    # Save after phase 1
    torch.save({
        "model_state_dict": model.state_dict(),
        "protopnet_accuracy": protopnet_acc,
        "num_prototypes_per_class": args.num_prototypes_per_class,
        "label_to_idx": label_to_idx,
    }, save_dir / "phase1_model.pth")

    # Phase 2: Train gating network
    final_acc = train_phase2(
        model, train_loader, val_loader, prior_model, device,
        epochs=args.phase2_epochs, lr=args.phase2_lr,
        variance_lambda=args.variance_lambda, use_wandb=use_wandb
    )

    # Get final w statistics from validation
    final_metrics = evaluate(model, val_loader, prior_model, device, use_fusion=True)

    # Save final model
    torch.save({
        "model_state_dict": model.state_dict(),
        "protopnet_accuracy": protopnet_acc,
        "final_accuracy": final_acc,
        "final_temperature": model.temperature.item(),
        "final_epsilon": model.epsilon.item(),
        "final_w_mean": final_metrics['w_mean'],
        "final_w_std": final_metrics['w_std'],
        "num_prototypes_per_class": args.num_prototypes_per_class,
        "label_to_idx": label_to_idx,
    }, save_dir / "final_model.pth")

    # Print final results
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Phase 1 (ProtoPNet Head) Accuracy: {100*protopnet_acc:.2f}%")
    print(f"Phase 2 (Fused Model)    Accuracy: {100*final_acc:.2f}%")
    print(f"\nProtoPNet Configuration:")
    print(f"  prototypes_per_class:  {args.num_prototypes_per_class}")
    print(f"  total_prototypes:      {model.classifier.num_prototypes}")
    print(f"\nFinal Learned Parameters:")
    print(f"  temperature:           {model.temperature.item():.4f}")
    print(f"  epsilon:               {model.epsilon.item():.6f}")
    print(f"\nGating Weight Statistics:")
    print(f"  w mean: {final_metrics['w_mean']:.4f}")
    print(f"  w std:  {final_metrics['w_std']:.4f}")
    print(f"  w min:  {final_metrics['w_min']:.4f}")
    print(f"  w max:  {final_metrics['w_max']:.4f}")
    print(f"\nModel saved to: {save_dir / 'final_model.pth'}")

    if use_wandb:
        wandb.log({
            "final/protopnet_accuracy": protopnet_acc,
            "final/fused_accuracy": final_acc,
            "final/temperature": model.temperature.item(),
            "final/epsilon": model.epsilon.item(),
            "final/w_mean": final_metrics['w_mean'],
            "final/w_std": final_metrics['w_std'],
            "final/w_min": final_metrics['w_min'],
            "final/w_max": final_metrics['w_max'],
            "final/num_prototypes_per_class": args.num_prototypes_per_class,
            "final/total_prototypes": model.classifier.num_prototypes,
        })
        wandb.finish()


if __name__ == "__main__":
    main()
