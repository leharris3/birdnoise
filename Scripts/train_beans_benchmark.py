"""
Train linear classifier on NatureLM-audio encoder for BEANS benchmark.

Evaluates both:
1. Likelihood model: p(species|audio)
2. Posterior model: p(species|audio) * p(species|space,time)

Metrics: Probe (accuracy), R-AUC (retrieval), NMI (clustering)
"""

import argparse
import warnings
import sys
import os
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import soundfile as sf
import resampy
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    normalized_mutual_info_score,
    top_k_accuracy_score
)
from sklearn.preprocessing import label_binarize
from scipy.spatial.distance import cdist

# Suppress audio warnings
warnings.filterwarnings("ignore", message=".*Xing stream size.*")
warnings.filterwarnings("ignore", message=".*Illegal Audio-MPEG-Header.*")
warnings.filterwarnings("ignore", category=UserWarning)

# Add NatureLM to path
sys.path.insert(0, str(Path(__file__).parent.parent / "NatureLM-audio"))

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("Warning: wandb not installed, logging disabled")

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed, visualizations disabled")


# ============================================================================
# Dataset Classes
# ============================================================================

class CBIBirdDataset(Dataset):
    """CBI dataset for linear classifier training."""
    
    def __init__(
        self,
        audio_dir: Path,
        metadata_df: pd.DataFrame,
        label_to_idx: dict,
        max_length_seconds: float = 10.0,
        sample_rate: int = 16000,
    ):
        self.audio_dir = Path(audio_dir)
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.max_length_seconds = max_length_seconds
        self.sample_rate = sample_rate
        self.max_samples = int(max_length_seconds * sample_rate)
        
    def __len__(self):
        return len(self.metadata_df)
    
    def __getitem__(self, idx):
        row = self.metadata_df.iloc[idx]
        
        # Load audio
        # Handle subdirectory structure: train_audio/{species}/{filename}
        species_code = row["ebird_code"]
        filename = row["filename"]
        audio_path = self.audio_dir / species_code / filename
        
        try:
            audio, sr = sf.read(audio_path)
        except Exception as e:
            # Return zeros on error
            audio = np.zeros(self.max_samples, dtype=np.float32)
            sr = self.sample_rate
        
        # Convert to mono if stereo
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
        
        # Get label
        label = self.label_to_idx[row["species"]]
        
        # Get metadata for prior
        metadata = {
            "latitude": row.get("latitude", np.nan),
            "longitude": row.get("longitude", np.nan),
            "date": row.get("date", ""),
            "filename": filename,
        }
        
        return {
            "audio": torch.from_numpy(audio.astype(np.float32)),
            "label": label,
            "metadata": metadata,
        }


class BEANSTestDataset(Dataset):
    """BEANS-Zero CBI test dataset with matched metadata."""
    
    def __init__(
        self,
        beans_subset,
        original_metadata_df: pd.DataFrame,
        label_to_idx: dict,
        sample_rate: int = 16000,
    ):
        self.beans_data = list(beans_subset)
        self.label_to_idx = label_to_idx
        self.sample_rate = sample_rate
        
        # Create filename -> metadata mapping
        self.metadata_map = {}
        for _, row in original_metadata_df.iterrows():
            xc_id = row["filename"].replace(".mp3", "")
            self.metadata_map[xc_id] = {
                "latitude": row.get("latitude", np.nan),
                "longitude": row.get("longitude", np.nan),
                "date": row.get("date", ""),
            }
    
    def __len__(self):
        return len(self.beans_data)
    
    def __getitem__(self, idx):
        sample = self.beans_data[idx]
        
        # Audio is already a numpy array in BEANS
        audio = np.array(sample["audio"], dtype=np.float32)
        
        # Get original sample rate from metadata
        meta = json.loads(sample["metadata"])
        orig_sr = int(meta.get("sample_rate", 44100))
        
        # Resample if needed
        if orig_sr != self.sample_rate:
            audio = resampy.resample(audio, orig_sr, self.sample_rate)
        
        # Pad/truncate to 10 seconds
        max_samples = self.sample_rate * 10
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        elif len(audio) < max_samples:
            audio = np.pad(audio, (0, max_samples - len(audio)))
        
        # Get label
        label_name = sample["output"]
        label = self.label_to_idx.get(label_name, -1)
        
        # Get matched metadata from original CBI
        xc_id = sample["file_name"].replace(".wav", "")
        metadata = self.metadata_map.get(xc_id, {
            "latitude": np.nan,
            "longitude": np.nan,
            "date": "",
        })
        metadata["filename"] = sample["file_name"]
        
        return {
            "audio": torch.from_numpy(audio),
            "label": label,
            "metadata": metadata,
        }


# ============================================================================
# Model Classes
# ============================================================================

class NatureLMLinearClassifier(nn.Module):
    """Linear classifier on top of frozen NatureLM encoder."""
    
    def __init__(self, num_classes: int, use_qformer: bool = True, pooling: str = "mean"):
        super().__init__()
        self.use_qformer = use_qformer
        self.pooling = pooling
        
        # Load NatureLM
        print("Loading NatureLM-audio...")
        from NatureLM.models import NatureLM
        self.naturelm = NatureLM.from_pretrained("EarthSpeciesProject/NatureLM-audio")
        
        # Freeze encoder
        for param in self.naturelm.parameters():
            param.requires_grad = False
        self.naturelm.eval()
        
        # Determine encoder dimension based on model configuration
        # NatureLM's encode_audio output dimension depends on:
        # - If use_audio_Qformer: outputs are projected to llama hidden size
        # - If max_pooling: outputs are projected to llama hidden size
        # - Otherwise: raw BEATs output (768)
        if self.naturelm.use_audio_Qformer or self.naturelm.max_pooling:
            self.encoder_dim = self.naturelm.llama_model.config.hidden_size
        else:
            self.encoder_dim = self.naturelm.beats.cfg.encoder_embed_dim
        
        print(f"Encoder dimension: {self.encoder_dim}")
        
        # Linear classifier
        self.classifier = nn.Linear(self.encoder_dim, num_classes)
    
    @torch.no_grad()
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract features from audio.
        
        NatureLM's encode_audio() already handles:
        - BEATs encoding
        - Q-Former processing (if use_audio_Qformer=True)
        - Projection to LLM space
        
        Returns embeddings ready for classification.
        """
        self.naturelm.eval()
        
        # encode_audio returns (audio_embeds, audio_atts)
        # audio_embeds is already processed through Q-Former and projected if enabled
        audio_embeds, audio_atts = self.naturelm.encode_audio(audio)
        
        # Pool over sequence dimension
        if self.pooling == "mean":
            # Use attention mask for proper mean
            mask = audio_atts.unsqueeze(-1).float()
            features = (audio_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        elif self.pooling == "max":
            features = audio_embeds.max(dim=1)[0]
        else:  # cls - use first token
            features = audio_embeds[:, 0, :]
        
        return features
    
    def forward(self, audio: torch.Tensor) -> tuple:
        """Forward pass returning logits and features."""
        features = self.encode(audio)
        logits = self.classifier(features)
        return logits, features


# ============================================================================
# Prior Model
# ============================================================================

class EBirdPriorWrapper:
    """Wrapper for eBird prior model with proper species code mapping."""
    
    def __init__(self, priors_dir: Path, species_list: list, cbi_metadata_path: Path):
        from eBirdPrior import EBirdCOGPrior
        
        self.prior = EBirdCOGPrior(priors_dir)
        self.species_list = species_list
        
        # Load species name -> ebird_code mapping from CBI metadata
        cbi_df = pd.read_csv(cbi_metadata_path)
        species_to_code_df = cbi_df[['species', 'ebird_code']].drop_duplicates()
        self.species_to_code = dict(zip(species_to_code_df['species'], species_to_code_df['ebird_code']))
        
        # Check which eBird codes actually have prior files
        priors_dir_path = Path(priors_dir)
        self.codes_with_prior = set()
        for code in self.species_to_code.values():
            tif_path = priors_dir_path / f"{code}_abundance_27km_uint8_cog.tif"
            if tif_path.exists():
                self.codes_with_prior.add(code)
        
        # Track which species we can look up
        self.valid_species = set()
        for species in species_list:
            code = self.species_to_code.get(species)
            if code and code in self.codes_with_prior:
                self.valid_species.add(species)
        
        missing_codes = set(self.species_to_code.values()) - self.codes_with_prior
        if missing_codes:
            missing_species = [s for s, c in self.species_to_code.items() if c in missing_codes]
            print(f"⚠️  Missing prior files for {len(missing_codes)} species: {missing_codes}")
            print(f"   Species: {missing_species}")
        
        print(f"Prior wrapper: {len(self.valid_species)}/{len(species_list)} species have prior files")
        
    def get_prior_probs(
        self, 
        latitudes: list, 
        longitudes: list, 
        dates: list
    ) -> np.ndarray:
        """
        Get prior probabilities for all species given locations and dates.
        Uses BATCH lookups for speed (56000x faster than sequential!).
        
        Returns: (batch_size, num_species) array of prior probabilities
        """
        batch_size = len(latitudes)
        num_species = len(self.species_list)
        
        # Start with uniform prior (neutral - doesn't change argmax)
        uniform_prob = 1.0 / num_species
        prior_probs = np.ones((batch_size, num_species)) * uniform_prob
        
        # Convert latitudes/longitudes to float arrays
        lat_arr = np.zeros(batch_size)
        lon_arr = np.zeros(batch_size)
        valid_mask = np.zeros(batch_size, dtype=bool)
        
        for i, (lat, lon) in enumerate(zip(latitudes, longitudes)):
            try:
                lat_f = float(lat) if lat is not None else np.nan
                lon_f = float(lon) if lon is not None else np.nan
                if not (pd.isna(lat_f) or pd.isna(lon_f)):
                    lat_arr[i] = lat_f
                    lon_arr[i] = lon_f
                    valid_mask[i] = True
            except (ValueError, TypeError):
                pass
        
        # Convert dates to week indices
        week_indices = np.zeros(batch_size, dtype=int)
        for i, date_str in enumerate(dates):
            try:
                date = pd.to_datetime(date_str)
                week_indices[i] = date.isocalendar()[1]
            except:
                week_indices[i] = 26  # Mid-year default
        
        # Build coords array for valid points
        coords = np.column_stack([lat_arr, lon_arr])
        
        # Group samples by week for efficient batch lookups
        unique_weeks = np.unique(week_indices)
        
        num_valid_lookups = 0
        num_failed = 0
        
        for week in unique_weeks:
            week_mask = (week_indices == week) & valid_mask
            if not week_mask.any():
                continue
            
            week_coords = coords[week_mask]
            week_indices_local = np.where(week_mask)[0]
            
            # Batch lookup for each species at this week
            for j, species in enumerate(self.species_list):
                ebird_code = self.species_to_code.get(species)
                if ebird_code is None or ebird_code not in self.codes_with_prior:
                    # Skip species without prior files - will keep uniform prior
                    continue
                
                try:
                    # Use batch lookup - MUCH faster!
                    # Ensure coords are in correct format: (N, 2) with [lat, lon]
                    if week_coords.shape[1] != 2:
                        raise ValueError(f"Expected coords shape (N, 2), got {week_coords.shape}")
                    
                    probs = self.prior.probs_batch(ebird_code, week_coords, week_idx=int(week), method="point")
                    
                    # probs_batch returns 1D array of shape (N,)
                    if not isinstance(probs, np.ndarray):
                        probs = np.array(probs)
                    if len(probs.shape) > 1:
                        probs = probs.flatten()
                    
                    # Ensure lengths match
                    if len(probs) != len(week_indices_local):
                        raise ValueError(f"Probs length {len(probs)} != coords length {len(week_indices_local)}")
                    
                    # Assign to correct positions
                    # Note: Zero probabilities are valid (species not present), but we only update if > 0
                    # to avoid overwriting uniform with zeros
                    for local_i, global_i in enumerate(week_indices_local):
                        if probs[local_i] > 0:
                            prior_probs[global_i, j] = probs[local_i]
                            num_valid_lookups += 1
                except Exception as e:
                    num_failed += 1
                    # Log first few errors for debugging
                    if num_failed <= 3:
                        print(f"      Warning: Prior lookup failed for {species} ({ebird_code}): {type(e).__name__}: {e}")
                    pass  # Keep uniform prior if lookup fails
        
        # Count successful lookups (non-zero probabilities found)
        # Note: Zero probabilities are valid (species not present), but we only count non-zero as "success"
        total_possible = batch_size * len([s for s in self.species_list if s in self.valid_species])
        success_rate = num_valid_lookups / total_possible if total_possible > 0 else 0
        print(f"    Prior: {num_valid_lookups}/{total_possible} non-zero probabilities found ({100*success_rate:.1f}%), {num_failed} lookup errors")
        
        # Normalize each row
        row_sums = prior_probs.sum(axis=1, keepdims=True)
        prior_probs = prior_probs / row_sums
        
        return prior_probs


# ============================================================================
# Metrics
# ============================================================================

def compute_probe_accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    """Compute linear probe accuracy."""
    preds = logits.argmax(axis=1)
    return accuracy_score(labels, preds)


def compute_retrieval_auc(features: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute retrieval ROC-AUC using cosine similarity.
    For each query, rank all other samples by similarity.
    """
    # Normalize features
    features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    
    # Compute pairwise cosine similarities
    similarities = features_norm @ features_norm.T
    
    # Create binary labels: 1 if same class, 0 otherwise
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


def compute_nmi(features: np.ndarray, labels: np.ndarray, n_clusters: int = None) -> float:
    """
    Compute Normalized Mutual Information using k-means clustering.
    """
    from sklearn.cluster import KMeans
    
    if n_clusters is None:
        n_clusters = len(np.unique(labels))
    
    # Normalize features
    features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=min(n_clusters, len(features)), random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_norm)
    
    return normalized_mutual_info_score(labels, cluster_labels)


def compute_all_metrics(
    logits: np.ndarray, 
    features: np.ndarray, 
    labels: np.ndarray,
    prior_probs: np.ndarray = None,
) -> dict:
    """Compute all metrics for likelihood and posterior models."""
    
    metrics = {}
    
    # Likelihood metrics
    metrics["probe_acc"] = compute_probe_accuracy(logits, labels)
    metrics["retrieval_auc"] = compute_retrieval_auc(features, labels)
    metrics["nmi"] = compute_nmi(features, labels)
    
    # Posterior metrics (if prior provided)
    if prior_probs is not None:
        # Convert logits to probabilities
        likelihood_probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        
        # Check if prior is informative (not uniform)
        prior_entropy = -np.sum(prior_probs * np.log(prior_probs + 1e-10), axis=1)
        max_entropy = np.log(len(prior_probs[0]))  # Uniform entropy
        prior_info = 1 - (prior_entropy / max_entropy).mean()
        
        # Posterior = likelihood * prior (normalized)
        # Use log-space for numerical stability
        log_likelihood = np.log(likelihood_probs + 1e-10)
        log_prior = np.log(prior_probs + 1e-10)
        
        # Combine in log space: log(posterior) = log(likelihood) + log(prior) - log(normalizer)
        log_posterior = log_likelihood + log_prior
        # Normalize (subtract log-sum-exp for numerical stability)
        log_posterior_max = log_posterior.max(axis=1, keepdims=True)
        log_posterior = log_posterior - log_posterior_max
        posterior_probs = np.exp(log_posterior)
        posterior_probs = posterior_probs / posterior_probs.sum(axis=1, keepdims=True)
        posterior_logits = np.log(posterior_probs + 1e-10)
        
        metrics["posterior_probe_acc"] = compute_probe_accuracy(posterior_logits, labels)
        metrics["prior_info"] = prior_info  # How informative the prior is (0=uniform, 1=perfect)
        # R-AUC and NMI use features, not affected by prior
        metrics["posterior_retrieval_auc"] = metrics["retrieval_auc"]
        metrics["posterior_nmi"] = metrics["nmi"]
    
    return metrics


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_top_species_comparison(
    likelihood_probs: np.ndarray,
    prior_probs: np.ndarray,
    posterior_probs: np.ndarray,
    true_label: int,
    idx_to_label: dict,
    sample_idx: int,
    metadata: dict = None,
    top_k: int = 10,
) -> plt.Figure:
    """
    Create bar chart comparing top-K species from likelihood, prior, and posterior.
    
    Args:
        likelihood_probs: (num_species,) array of likelihood probabilities
        prior_probs: (num_species,) array of prior probabilities
        posterior_probs: (num_species,) array of posterior probabilities
        true_label: True species label index
        idx_to_label: Mapping from index to species name
        sample_idx: Sample index for title
        metadata: Dict with 'latitude', 'longitude', 'date' for prior plot
        top_k: Number of top species to show
    """
    if not HAS_MATPLOTLIB:
        return None
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Sample {sample_idx}: Top {top_k} Species Probabilities', fontsize=14, fontweight='bold')
    
    # Get top-K indices for each
    top_likelihood = np.argsort(likelihood_probs)[-top_k:][::-1]
    top_prior = np.argsort(prior_probs)[-top_k:][::-1]
    top_posterior = np.argsort(posterior_probs)[-top_k:][::-1]
    
    # Get all unique species to show
    all_top = set(top_likelihood) | set(top_prior) | set(top_posterior)
    if true_label not in all_top:
        all_top.add(true_label)
    
    all_top = sorted(list(all_top), key=lambda x: posterior_probs[x], reverse=True)[:top_k]
    
    species_names = [idx_to_label[i] for i in all_top]
    likelihood_vals = [likelihood_probs[i] for i in all_top]
    prior_vals = [prior_probs[i] for i in all_top]
    posterior_vals = [posterior_probs[i] for i in all_top]
    
    # Colors: highlight true species
    colors_likelihood = ['#2ecc71' if i == true_label else '#3498db' for i in all_top]
    colors_prior = ['#2ecc71' if i == true_label else '#e74c3c' for i in all_top]
    colors_posterior = ['#2ecc71' if i == true_label else '#9b59b6' for i in all_top]
    
    # Plot likelihood
    axes[0].barh(range(len(species_names)), likelihood_vals, color=colors_likelihood, alpha=0.7)
    axes[0].set_yticks(range(len(species_names)))
    axes[0].set_yticklabels(species_names, fontsize=9)
    axes[0].set_xlabel('Probability', fontsize=10)
    axes[0].set_title('Likelihood\np(species|audio)', fontsize=11, fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', alpha=0.3)
    
    # Plot prior
    title_prior = 'Prior\np(species|space,time)'
    if metadata:
        lat = metadata.get('latitude', None)
        lon = metadata.get('longitude', None)
        date = metadata.get('date', 'N/A')
        if lat is not None and lon is not None:
            try:
                lat_f = float(lat)
                lon_f = float(lon)
                title_prior += f'\n(lat={lat_f:.2f}, lon={lon_f:.2f}, date={date})'
            except (ValueError, TypeError):
                title_prior += f'\n(lat=N/A, lon=N/A, date={date})'
        else:
            title_prior += f'\n(lat=N/A, lon=N/A, date={date})'
    axes[1].barh(range(len(species_names)), prior_vals, color=colors_prior, alpha=0.7)
    axes[1].set_yticks(range(len(species_names)))
    axes[1].set_yticklabels(species_names, fontsize=9)
    axes[1].set_xlabel('Probability', fontsize=10)
    axes[1].set_title(title_prior, fontsize=11, fontweight='bold')
    axes[1].invert_yaxis()
    axes[1].grid(axis='x', alpha=0.3)
    
    # Plot posterior
    axes[2].barh(range(len(species_names)), posterior_vals, color=colors_posterior, alpha=0.7)
    axes[2].set_yticks(range(len(species_names)))
    axes[2].set_yticklabels(species_names, fontsize=9)
    axes[2].set_xlabel('Probability', fontsize=10)
    axes[2].set_title('Posterior\np(species|audio,space,time)', fontsize=11, fontweight='bold')
    axes[2].invert_yaxis()
    axes[2].grid(axis='x', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', alpha=0.7, label='True Species'),
        Patch(facecolor='#3498db', alpha=0.7, label='Other Species (Likelihood)'),
        Patch(facecolor='#e74c3c', alpha=0.7, label='Other Species (Prior)'),
        Patch(facecolor='#9b59b6', alpha=0.7, label='Other Species (Posterior)'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=9)
    
    plt.tight_layout()
    return fig


def log_sample_predictions(
    logits: np.ndarray,
    prior_probs: np.ndarray,
    labels: np.ndarray,
    idx_to_label: dict,
    metadata_list: list,
    epoch: int,
    split: str,
    num_samples: int = 3,
    prior_model: EBirdPriorWrapper = None,
):
    """
    Log sample predictions to wandb with visualizations.
    
    Args:
        logits: (batch_size, num_species) array of logits
        prior_probs: (batch_size, num_species) array of prior probabilities or None
        labels: (batch_size,) array of true labels
        idx_to_label: Mapping from index to species name
        metadata_list: List of metadata dicts with lat/lon/date
        epoch: Current epoch
        split: 'train' or 'val'
        num_samples: Number of samples to visualize
        prior_model: Prior model wrapper (for getting prior if not provided)
    """
    if not HAS_WANDB or not HAS_MATPLOTLIB:
        return
    
    # Convert logits to probabilities
    likelihood_probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    
    # Compute posterior if prior provided
    if prior_probs is not None:
        posterior_probs = likelihood_probs * prior_probs
        posterior_probs = posterior_probs / posterior_probs.sum(axis=1, keepdims=True)
    else:
        posterior_probs = likelihood_probs
        prior_probs = np.ones_like(likelihood_probs) / likelihood_probs.shape[1]  # Uniform
    
    # Select random samples to visualize
    batch_size = len(labels)
    sample_indices = np.random.choice(batch_size, size=min(num_samples, batch_size), replace=False)
    
    plots = []
    for i, sample_idx in enumerate(sample_indices):
        metadata = metadata_list[sample_idx] if sample_idx < len(metadata_list) else {}
        
        fig = plot_top_species_comparison(
            likelihood_probs=likelihood_probs[sample_idx],
            prior_probs=prior_probs[sample_idx],
            posterior_probs=posterior_probs[sample_idx],
            true_label=labels[sample_idx],
            idx_to_label=idx_to_label,
            sample_idx=sample_idx,
            metadata=metadata,
            top_k=10,
        )
        
        if fig is not None:
            plots.append(wandb.Image(fig, caption=f"{split}_epoch{epoch}_sample{i}"))
            plt.close(fig)
    
    if plots:
        wandb.log({f"{split}/species_predictions": plots}, step=epoch)


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    device: str,
    collect_viz_sample: bool = False,
) -> dict:
    """Train for one epoch."""
    model.classifier.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    # Collect a small sample for visualization (once per epoch)
    viz_logits = []
    viz_labels = []
    viz_metadata = []
    viz_collected = False
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        audio = batch["audio"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        
        with autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(audio)
            loss = F.cross_entropy(logits, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        total_loss += loss.item() * len(labels)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += len(labels)
        
        # Collect sample for visualization (first batch only)
        if collect_viz_sample and not viz_collected:
            viz_logits.append(logits.float().cpu().detach())
            viz_labels.append(labels.cpu().detach())
            # Extract metadata
            batch_meta = batch["metadata"]
            for i in range(len(labels)):
                if isinstance(batch_meta["latitude"], list):
                    viz_metadata.append({
                        "latitude": batch_meta["latitude"][i],
                        "longitude": batch_meta["longitude"][i],
                        "date": batch_meta["date"][i] if i < len(batch_meta["date"]) else "",
                    })
                else:
                    viz_metadata.append({
                        "latitude": batch_meta["latitude"],
                        "longitude": batch_meta["longitude"],
                        "date": batch_meta["date"],
                    })
            viz_collected = True
        
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100*correct/total:.2f}%")
    
    result = {
        "loss": total_loss / total,
        "accuracy": correct / total,
    }
    
    # Add visualization data if collected
    if collect_viz_sample and viz_logits:
        result["_viz_data"] = {
            "logits": torch.cat(viz_logits, dim=0).detach().numpy(),
            "labels": torch.cat(viz_labels, dim=0).detach().numpy(),
            "metadata_list": viz_metadata,
        }
    
    return result


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    prior_model: EBirdPriorWrapper = None,
) -> dict:
    """Evaluate model and compute all metrics."""
    model.eval()
    
    all_logits = []
    all_features = []
    all_labels = []
    all_metadata = []
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        audio = batch["audio"].to(device)
        labels = batch["label"]
        
        with autocast("cuda", dtype=torch.bfloat16):
            logits, features = model(audio)
        
        all_logits.append(logits.float().cpu().detach())
        all_features.append(features.float().cpu().detach())
        all_labels.append(labels.detach() if isinstance(labels, torch.Tensor) else labels)
        all_metadata.extend([batch["metadata"]])
    
    logits = torch.cat(all_logits, dim=0).detach().numpy()
    features = torch.cat(all_features, dim=0).detach().numpy()
    labels = torch.cat(all_labels, dim=0)
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().numpy()
    else:
        labels = np.array(labels)
    
    # Compute prior if available
    prior_probs = None
    if prior_model is not None:
        print("  Computing space-time prior...")
        # Extract location/time from metadata
        latitudes = []
        longitudes = []
        dates = []
        for meta_batch in all_metadata:
            batch_lats = meta_batch["latitude"]
            batch_lons = meta_batch["longitude"]
            batch_dates = meta_batch["date"]
            
            if isinstance(batch_lats, list):
                latitudes.extend(batch_lats)
                longitudes.extend(batch_lons)
                dates.extend(batch_dates)
            else:
                latitudes.append(batch_lats)
                longitudes.append(batch_lons)
                dates.append(batch_dates)
        
        latitudes = latitudes[:len(labels)]
        longitudes = longitudes[:len(labels)]
        dates = dates[:len(labels)]
        
        prior_probs = prior_model.get_prior_probs(latitudes, longitudes, dates)
        print(f"  Prior computed for {len(labels)} samples")
    
    # Flatten metadata list for visualization
    metadata_list = []
    for meta_batch in all_metadata:
        batch_lats = meta_batch["latitude"]
        batch_lons = meta_batch["longitude"]
        batch_dates = meta_batch["date"]
        
        if isinstance(batch_lats, list):
            for i in range(len(batch_lats)):
                metadata_list.append({
                    "latitude": batch_lats[i],
                    "longitude": batch_lons[i],
                    "date": batch_dates[i] if i < len(batch_dates) else "",
                })
        else:
            metadata_list.append({
                "latitude": batch_lats,
                "longitude": batch_lons,
                "date": batch_dates,
            })
    
    # Ensure metadata_list matches labels length
    metadata_list = metadata_list[:len(labels)]
    
    metrics = compute_all_metrics(logits, features, labels, prior_probs)
    
    # Add visualization data to metrics
    metrics["_viz_data"] = {
        "logits": logits,
        "prior_probs": prior_probs,
        "labels": labels,
        "metadata_list": metadata_list,
    }
    
    return metrics


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train BEANS benchmark linear classifier")
    
    # Data
    parser.add_argument("--data_dir", type=str, default="../Data/cbi",
                        help="Path to CBI data directory")
    parser.add_argument("--priors_dir", type=str, default="../Data/priors",
                        help="Path to eBird priors directory")
    
    # Model
    parser.add_argument("--use_qformer", action="store_true",
                        help="Use Q-Former (full NatureLM) vs just BEATs")
    parser.add_argument("--pooling", type=str, default="mean",
                        choices=["mean", "max", "cls"])
    
    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Validation split ratio from training data")
    
    # Hardware
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    
    # Logging
    parser.add_argument("--wandb_project", type=str, default="beans-benchmark")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    
    # Checkpointing
    parser.add_argument("--save_dir", type=str, default="../checkpoints_beans")
    parser.add_argument("--resume", type=str, default=None)
    
    # Options
    parser.add_argument("--use_prior", action="store_true",
                        help="Enable eBird prior for posterior evaluation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true",
                        help="Quick test run")
    
    return parser.parse_args()


def collate_fn(batch):
    """Custom collate to handle metadata."""
    audio = torch.stack([b["audio"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch])
    
    metadata = {
        "latitude": [b["metadata"]["latitude"] for b in batch],
        "longitude": [b["metadata"]["longitude"] for b in batch],
        "date": [b["metadata"]["date"] for b in batch],
        "filename": [b["metadata"]["filename"] for b in batch],
    }
    
    return {"audio": audio, "label": labels, "metadata": metadata}


def main():
    args = parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # ========== Load Data ==========
    print("\n" + "="*60)
    print("Loading data...")
    print("="*60)
    
    data_dir = Path(args.data_dir).resolve()
    train_csv = pd.read_csv(data_dir / "train.csv")
    
    # Load BEANS-Zero CBI test set
    print("Loading BEANS-Zero CBI test set...")
    from datasets import load_dataset
    ds = load_dataset("EarthSpeciesProject/BEANS-Zero", split="test")
    cbi_idx = np.where(np.array(ds["dataset_name"]) == "cbi")[0]
    beans_test = ds.select(cbi_idx)
    print(f"BEANS test samples: {len(beans_test)}")
    
    # Get BEANS test file IDs to exclude from training
    beans_test_files = set(s["file_name"].replace(".wav", "") for s in beans_test)
    
    # Filter training data (exclude BEANS test files)
    train_csv["xc_id"] = train_csv["filename"].str.replace(".mp3", "")
    train_mask = ~train_csv["xc_id"].isin(beans_test_files)
    train_df = train_csv[train_mask].copy()
    print(f"Training samples (after excluding BEANS test): {len(train_df)}")
    
    # Create label mapping from training data
    all_species = sorted(train_df["species"].unique())
    label_to_idx = {s: i for i, s in enumerate(all_species)}
    idx_to_label = {i: s for s, i in label_to_idx.items()}
    num_classes = len(all_species)
    print(f"Number of classes: {num_classes}")
    
    # Split into train/val
    train_split, val_split = train_test_split(
        train_df, test_size=args.val_split, stratify=train_df["species"], random_state=args.seed
    )
    print(f"Train split: {len(train_split)}, Val split: {len(val_split)}")
    
    # Create datasets
    audio_dir = data_dir / "train_audio"
    
    train_dataset = CBIBirdDataset(audio_dir, train_split, label_to_idx)
    val_dataset = CBIBirdDataset(audio_dir, val_split, label_to_idx)
    test_dataset = BEANSTestDataset(beans_test, train_csv, label_to_idx)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    # ========== Initialize Model ==========
    print("\n" + "="*60)
    print("Initializing model...")
    print("="*60)
    
    model = NatureLMLinearClassifier(
        num_classes=num_classes,
        use_qformer=args.use_qformer,
        pooling=args.pooling,
    )
    model = model.to(device)
    
    # ========== Initialize Prior (if enabled) ==========
    prior_model = None
    if args.use_prior:
        print("\nInitializing eBird prior...")
        try:
            prior_model = EBirdPriorWrapper(
                priors_dir=Path(args.priors_dir).resolve(),
                species_list=list(idx_to_label.values()),
                cbi_metadata_path=data_dir / "train.csv",
            )
            print("Prior model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load prior model: {e}")
            prior_model = None
    
    # ========== Optimizer & Scheduler ==========
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler("cuda")
    
    # ========== Resume if specified ==========
    start_epoch = 1
    best_val_acc = 0.0
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.classifier.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")
    
    # ========== Wandb ==========
    if HAS_WANDB and not args.no_wandb:
        run_name = args.wandb_run_name or f"beans_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
    
    # ========== Dry run check ==========
    if args.dry_run:
        print("\n" + "="*60)
        print("DRY RUN - Testing one batch...")
        print("="*60)
        batch = next(iter(train_loader))
        audio = batch["audio"].to(device)
        with autocast("cuda", dtype=torch.bfloat16):
            logits, features = model(audio)
        print(f"Logits shape: {logits.shape}")
        print(f"Features shape: {features.shape}")
        print("Dry run successful!")
        return
    
    # ========== Training Loop ==========
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save label mapping
    torch.save({"label_to_idx": label_to_idx, "idx_to_label": idx_to_label}, 
               save_dir / "label_mapping.pth")
    
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print("="*60)
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, device,
            collect_viz_sample=(HAS_WANDB and not args.no_wandb)
        )
        
        # Validate
        val_metrics = evaluate(model, val_loader, device, prior_model)
        
        # Log visualizations (once per epoch)
        # Training visualization (no prior, just likelihood)
        if HAS_WANDB and not args.no_wandb and "_viz_data" in train_metrics:
            train_viz = train_metrics.pop("_viz_data")
            # Use uniform prior for training visualization
            uniform_prior = np.ones((len(train_viz["logits"]), len(idx_to_label))) / len(idx_to_label)
            log_sample_predictions(
                logits=train_viz["logits"],
                prior_probs=uniform_prior,
                labels=train_viz["labels"],
                idx_to_label=idx_to_label,
                metadata_list=train_viz["metadata_list"],
                epoch=epoch,
                split="train",
                num_samples=3,
                prior_model=None,
            )
        
        # Validation visualization (with prior)
        if HAS_WANDB and not args.no_wandb and "_viz_data" in val_metrics:
            viz_data = val_metrics.pop("_viz_data")  # Remove from metrics dict
            log_sample_predictions(
                logits=viz_data["logits"],
                prior_probs=viz_data["prior_probs"],
                labels=viz_data["labels"],
                idx_to_label=idx_to_label,
                metadata_list=viz_data["metadata_list"],
                epoch=epoch,
                split="val",
                num_samples=3,
                prior_model=prior_model,
            )
        
        # Log
        print(f"\nTrain Loss: {train_metrics['loss']:.4f}, Acc: {100*train_metrics['accuracy']:.2f}%")
        print(f"Val Probe: {100*val_metrics['probe_acc']:.2f}%, R-AUC: {val_metrics['retrieval_auc']:.4f}, NMI: {val_metrics['nmi']:.4f}")
        
        if prior_model:
            prior_info = val_metrics.get('prior_info', 0.0)
            print(f"Val Posterior Probe: {100*val_metrics['posterior_probe_acc']:.2f}% (Prior info: {prior_info:.3f})")
        
        if HAS_WANDB and not args.no_wandb:
            log_dict = {
                "epoch": epoch,
                "train/loss": train_metrics["loss"],
                "train/accuracy": train_metrics["accuracy"],
                "val/probe_acc": val_metrics["probe_acc"],
                "val/retrieval_auc": val_metrics["retrieval_auc"],
                "val/nmi": val_metrics["nmi"],
            }
            if prior_model:
                log_dict["val/posterior_probe_acc"] = val_metrics["posterior_probe_acc"]
                if "prior_info" in val_metrics:
                    log_dict["val/prior_info"] = val_metrics["prior_info"]
            wandb.log(log_dict)
        
        # Save best model
        if val_metrics["probe_acc"] > best_val_acc:
            best_val_acc = val_metrics["probe_acc"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.classifier.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": best_val_acc,
            }, save_dir / "best_model.pth")
            print(f"  New best model saved! (acc: {100*best_val_acc:.2f}%)")
        
        # Save periodic checkpoint
        if epoch % 5 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.classifier.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_metrics["probe_acc"],
            }, save_dir / f"checkpoint_epoch{epoch}.pth")
    
    # ========== Final Test Evaluation ==========
    print("\n" + "="*60)
    print("Final BEANS Test Evaluation")
    print("="*60)
    
    # Load best model
    checkpoint = torch.load(save_dir / "best_model.pth", map_location=device)
    model.classifier.load_state_dict(checkpoint["model_state_dict"])
    
    test_metrics = evaluate(model, test_loader, device, prior_model)
    
    print(f"\n{'='*60}")
    print("BEANS CBI Test Results (Table 8 comparison)")
    print("="*60)
    print(f"Probe Accuracy: {100*test_metrics['probe_acc']:.2f}%")
    print(f"Retrieval AUC:  {test_metrics['retrieval_auc']:.4f}")
    print(f"NMI:            {test_metrics['nmi']:.4f}")
    
    if prior_model:
        print(f"\n--- With Space-Time Prior ---")
        print(f"Posterior Probe: {100*test_metrics['posterior_probe_acc']:.2f}%")
    
    # Reference from paper (BEATs-NatureLM-audio row in Table 8)
    print(f"\n--- Paper Reference (BEATs-NatureLM-audio) ---")
    print(f"Probe: 0.359 (35.9%)")
    print(f"R-AUC: 0.711")
    
    if HAS_WANDB and not args.no_wandb:
        wandb.log({
            "test/probe_acc": test_metrics["probe_acc"],
            "test/retrieval_auc": test_metrics["retrieval_auc"],
            "test/nmi": test_metrics["nmi"],
        })
        if prior_model:
            wandb.log({"test/posterior_probe_acc": test_metrics["posterior_probe_acc"]})
        wandb.finish()
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()

