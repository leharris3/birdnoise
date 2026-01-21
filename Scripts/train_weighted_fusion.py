"""
Train learnable weighted fusion of audio classifier and space-time prior.

Model: final_logits = audio_logits / T + w(a,x,t) * log(prior_robust)
where:
- prior_robust = (1-eps)*prior_probs + eps*(1/K)
- w(a,x,t) is learned (scalar or gating network)
- T is learnable temperature
"""

import argparse
import warnings
import sys
import os
from pathlib import Path
from datetime import datetime
import json
import math

import librosa
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
from sklearn.metrics import accuracy_score, top_k_accuracy_score, confusion_matrix, roc_auc_score, normalized_mutual_info_score
from sklearn.cluster import KMeans

# Suppress audio warnings and MP3 decoding notes
warnings.filterwarnings("ignore", message=".*Xing stream size.*")
warnings.filterwarnings("ignore", message=".*Illegal Audio-MPEG-Header.*")
warnings.filterwarnings("ignore", message=".*trying to resync.*")
warnings.filterwarnings("ignore", message=".*hit end of available data.*")
warnings.filterwarnings("ignore", message=".*AUDIO-MPEG-HEADER.*")
warnings.filterwarnings("ignore", category=UserWarning)

# Suppress MP3 decoder notes (these come from mpg123 via soundfile)
import os
# Redirect stderr for soundfile operations (but keep it for other errors)
# We'll do this more selectively in the dataset class

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
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed, visualizations disabled")


# ============================================================================
# Dataset Classes
# ============================================================================

class CBIBirdDataset(Dataset):
    """CBI dataset with location/time metadata."""
    
    def __init__(
        self,
        audio_dir: Path,
        metadata_df: pd.DataFrame,
        label_to_idx: dict,
        max_length_seconds: float = 10.0,
        sample_rate: int = 16000,
    ):
        self.audio_dir = Path(audio_dir)
        
        # Keep original index for cache lookup
        self.metadata_df = metadata_df.copy()
        self.original_indices = self.metadata_df.index.values
        self.metadata_df = self.metadata_df.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.max_length_seconds = max_length_seconds
        self.sample_rate = sample_rate
        self.max_samples = int(max_length_seconds * sample_rate)
        
    def __len__(self):
        return len(self.metadata_df)
    
    def __getitem__(self, idx):

        row = self.metadata_df.iloc[idx]
        
        # Load audio
        species_code = row["ebird_code"]
        filename = row["filename"]

        import glob

        # match the first '.ogg' or '.mp3' in audio_files
        audio_path = self.audio_dir / species_code / filename

        # HACK: .mp3 -> .ogg
        audio_path = audio_path.__str__().replace(".mp3", ".ogg") 
        
        try:
            # Suppress MP3 decoder notes (these come from mpg123 C library via soundfile)
            # We'll redirect stderr temporarily and filter out the harmless MP3 notes
            import sys
            from io import StringIO
            
            class FilteredStderr:
                """Filter stderr to suppress MP3 decoder notes."""
                def __init__(self, original_stderr):
                    self.original_stderr = original_stderr
                    self.buffer = StringIO()
                
                def write(self, text):
                    # Filter out MP3 decoder notes
                    text_lower = text.lower()
                    if any(note in text_lower for note in [
                        'note:', 'trying to resync', 'hit end of available data',
                        'illegal audio-mpeg-header', 'xing stream size', 'dequantization failed'
                    ]):
                        return  # Suppress this message
                    # Pass through other messages
                    self.original_stderr.write(text)
                
                def flush(self):
                    self.original_stderr.flush()
            
            old_stderr = sys.stderr
            sys.stderr = FilteredStderr(old_stderr)
            try:
                audio, sr = sf.read(audio_path)
            finally:
                sys.stderr = old_stderr
        except Exception:

            # HACK
            # audio = np.zeros(self.max_samples, dtype=np.float32)
            # raise Exception(f"failed to load audio: {audio_path}")
            print(f"Error: failed to load:  {audio_path}")
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
        
        # Get metadata
        date_str = row.get("date", "")
        try:
            date = pd.to_datetime(date_str)
            day_of_year = date.timetuple().tm_yday
            hour = row.get("time", "12:00")
            if isinstance(hour, str) and ":" in hour:
                hour = int(hour.split(":")[0])
            else:
                hour = 12
        except:
            day_of_year = 182  # Mid-year
            hour = 12
        
        # Safely convert latitude/longitude
        def safe_float(val, default=0.0):
            if pd.isna(val):
                return default
            try:
                return float(val)
            except (ValueError, TypeError):
                return default
        
        metadata = {
            "latitude": safe_float(row.get("latitude"), 0.0),
            "longitude": safe_float(row.get("longitude"), 0.0),
            "day_of_year": day_of_year,
            "hour": hour,
            "date": date_str,
        }
        
        # Get original index for cache lookup
        original_idx = self.original_indices[idx]
        
        return {
            "audio": torch.from_numpy(audio.astype(np.float32)),
            "label": label,
            "metadata": metadata,
            "sample_idx": original_idx,  # For cache lookup
        }


# ============================================================================
# Model Classes
# ============================================================================

class NatureLMAudioEncoder(nn.Module):
    """Frozen NatureLM audio encoder."""
    
    def __init__(self, use_qformer: bool = True, pooling: str = "mean"):
        super().__init__()
        self.use_qformer = use_qformer
        self.pooling = pooling
        
        print("Loading NatureLM-audio...")
        from NatureLM.models import NatureLM
        self.naturelm = NatureLM.from_pretrained("EarthSpeciesProject/NatureLM-audio")
        
        # Freeze encoder
        for param in self.naturelm.parameters():
            param.requires_grad = False
            
        self.naturelm.eval()
        
        # Determine encoder dimension
        if self.naturelm.use_audio_Qformer or self.naturelm.max_pooling:
            self.encoder_dim = self.naturelm.llama_model.config.hidden_size
        else:
            self.encoder_dim = self.naturelm.beats.cfg.encoder_embed_dim
        
        print(f"Encoder dimension: {self.encoder_dim}")
    
    # NOTE: we don't EVER finetune the audio encoder
    @torch.no_grad()
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract features from audio."""
        self.naturelm.eval()
        # Ensure audio is on the same device as model
        if audio.device != next(self.naturelm.parameters()).device:
            audio = audio.to(next(self.naturelm.parameters()).device)
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            audio_embeds, audio_atts = self.naturelm.encode_audio(audio)
        
        # Pool over sequence dimension
        if self.pooling == "mean":
            mask = audio_atts.unsqueeze(-1).float()
            features = (audio_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        elif self.pooling == "max":
            features = audio_embeds.max(dim=1)[0]
        else:
            features = audio_embeds[:, 0, :]
        
        return features
    
    def forward(self, audio: torch.Tensor) -> tuple:
        """Forward pass returning logits and features."""
        features = self.encode(audio)
        return features


class PriorGatingNetwork(nn.Module):
    """
    Gating network to learn w(a,x,t) based on audio and prior features.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, w_max: float = 2.0):
        super().__init__()
        self.w_max = w_max
        
        # Input: audio features (max_prob, entropy, margin) + prior features + metadata
        # Metadata: sin/cos(day_of_year), sin/cos(hour), normalized lat, lon = 6 dims
        # Audio features: 3 dims
        # Prior features: 3 dims
        # Total: 12 dims
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, audio_features: torch.Tensor, prior_features: torch.Tensor, 
                metadata: dict) -> torch.Tensor:
        """
        Args:
            audio_features: (batch_size, 3) - [max_prob, entropy, margin]
            prior_features: (batch_size, 3) - [max_prob, entropy, margin]
            metadata: dict with 'day_of_year', 'hour', 'latitude', 'longitude'
        
        Returns:
            w: (batch_size,) - gate weights in [0, w_max]
        """
        batch_size = audio_features.size(0)
        
        # Extract metadata (ensure all are numeric)
        def safe_float_list(key, default=0.0):
            result = []
            for m in metadata:
                val = m.get(key, default)
                try:
                    result.append(float(val))
                except (ValueError, TypeError):
                    result.append(float(default))
            return result
        
        day_of_year = torch.tensor(safe_float_list("day_of_year", 182), 
                                   dtype=torch.float32, device=audio_features.device)
        hour = torch.tensor(safe_float_list("hour", 12), 
                           dtype=torch.float32, device=audio_features.device)
        lat = torch.tensor(safe_float_list("latitude", 0.0), 
                          dtype=torch.float32, device=audio_features.device)
        lon = torch.tensor(safe_float_list("longitude", 0.0), 
                          dtype=torch.float32, device=audio_features.device)
        
        # Normalize lat/lon (rough normalization to [-1, 1])
        lat_norm = lat / 90.0  # Latitude range: -90 to 90
        lon_norm = lon / 180.0  # Longitude range: -180 to 180
        
        # Sin/cos encoding for cyclical features
        day_sin = torch.sin(2 * math.pi * day_of_year / 365.25)
        day_cos = torch.cos(2 * math.pi * day_of_year / 365.25)
        hour_sin = torch.sin(2 * math.pi * hour / 24.0)
        hour_cos = torch.cos(2 * math.pi * hour / 24.0)
        
        # Concatenate all features
        meta_features = torch.stack([day_sin, day_cos, hour_sin, hour_cos, lat_norm, lon_norm], dim=1)
        combined = torch.cat([audio_features, prior_features, meta_features], dim=1)
        
        # MLP to gate logit
        gate_logit = self.mlp(combined).squeeze(-1)
        
        # Sigmoid and scale to [0, w_max]
        w = torch.sigmoid(gate_logit) * self.w_max
        
        return w


class WeightedFusionModel(nn.Module):
    """Model that fuses audio classifier with space-time prior."""
    
    def __init__(
        self,
        audio_encoder: NatureLMAudioEncoder,
        num_classes: int,
        use_gating: bool = False,
        w_max: float = 2.0,
        gate_hidden_dim: int = 64,
    ):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.num_classes = num_classes
        self.use_gating = use_gating
        
        # Audio classifier
        # p( y | x)
        self.audio_classifier = nn.Linear(audio_encoder.encoder_dim, num_classes)
        
        # Learnable parameters
        self.temperature = nn.Parameter(torch.ones(1))   # Temperature scaling
        self.epsilon = nn.Parameter(torch.tensor(0.01))  # Prior smoothing (clamped to [1e-8, 0.5])
        
        if use_gating:
            # Gating network
            self.gate_network = PriorGatingNetwork(
                input_dim=12,  # 3 audio + 3 prior + 6 metadata
                hidden_dim=gate_hidden_dim,
                w_max=w_max,
            )
            self.w_weight = None  # Not used when gating
        else:
            # Scalar weight
            self.w_weight = nn.Parameter(torch.tensor(0.5))  # Initial weight
            self.gate_network = None
    
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
    ) -> torch.Tensor:
        """
        Args:
            audio: (batch_size, audio_length)
            prior_probs: (batch_size, num_classes) - space-time prior probabilities
            metadata: List of metadata dicts for gating
        
        Returns:
            final_logits: (batch_size, num_classes)
        """
        # Audio logits
        features = self.audio_encoder(audio)
        audio_logits = self.audio_classifier(features)
        
        # Temperature scaling
        audio_logits_scaled = audio_logits / (self.temperature.abs() + 1e-8)
        
        # Robust prior: just add epsilon to avoid zeros (no mixing with uniform)
        # Raw abundance scores are already informative, so we don't need to dilute them
        eps = torch.clamp(self.epsilon, min=1e-8, max=0.1)  # Smaller max since we're not mixing
        prior_robust = prior_probs + eps  # Just add small constant to avoid log(0)
        
        # Log prior (model will learn scale via w and temperature)
        log_prior = torch.log(prior_robust + 1e-10)
        
        # Compute weight w(a,x,t)
        if self.use_gating and metadata is not None:
            audio_features = self.compute_audio_features(audio_logits_scaled)
            prior_features = self.compute_prior_features(prior_probs)
            w = self.gate_network(audio_features, prior_features, metadata)
        else:
            # Scalar weight (clamped to [0, w_max])
            w = torch.clamp(self.w_weight, min=0.0, max=2.0).expand(audio_logits.size(0))
        
        # Final logits: audio/T + w * log(prior)
        final_logits = audio_logits_scaled + w.unsqueeze(1) * log_prior
        
        return final_logits, audio_logits, prior_robust


# ============================================================================
# Season Mapping Functions
# ============================================================================

def week_to_season(week: int) -> int:
    """Map week (1-52) to season (0=Winter, 1=Spring, 2=Summer, 3=Fall)."""
    if week <= 13:
        return 0  # Winter (weeks 1-13)
    elif week <= 26:
        return 1  # Spring (weeks 14-26)
    elif week <= 39:
        return 2  # Summer (weeks 27-39)
    else:
        return 3  # Fall (weeks 40-52)

def season_to_weeks(season: int) -> list:
    """Map season to list of weeks."""
    if season == 0:  # Winter
        return list(range(1, 14))
    elif season == 1:  # Spring
        return list(range(14, 27))
    elif season == 2:  # Summer
        return list(range(27, 40))
    else:  # Fall
        return list(range(40, 53))

SEASON_NAMES = ["Winter", "Spring", "Summer", "Fall"]

# ============================================================================
# Prior Model with Caching
# ============================================================================

class EBirdPriorWrapper:
    """Wrapper for eBird prior model with caching and optional pre-computed cache."""
    
    def __init__(
        self, 
        priors_dir: Path, 
        species_list: list, 
        cbi_metadata_path: Path,
        cache_path: Path = None,
        df_index: pd.Index = None,
    ):
        """
        Args:
            priors_dir: Directory containing eBird prior TIF files
            species_list: List of species names
            cbi_metadata_path: Path to CBI metadata CSV
            cache_path: Optional path to pre-computed HDF5 cache
            df_index: Optional pandas Index matching the cache (if using cache)
        """
        self.species_list = species_list
        self.use_cache = cache_path is not None and Path(cache_path).exists()
        
        # Always initialize on-the-fly computation attributes (needed for fallback when sample_indices=None)
        from eBirdPrior import EBirdCOGPrior
        self.prior = EBirdCOGPrior(priors_dir)
        
        # Load species name -> ebird_code mapping
        cbi_df = pd.read_csv(cbi_metadata_path)
        species_to_code_df = cbi_df[['species', 'ebird_code']].drop_duplicates()
        self.species_to_code = dict(zip(species_to_code_df['species'], species_to_code_df['ebird_code']))
        
        # Check which codes have prior files
        priors_dir_path = Path(priors_dir)
        self.codes_with_prior = set()
        for code in self.species_to_code.values():
            tif_path = priors_dir_path / f"{code}_abundance_27km_uint8_cog.tif"
            if tif_path.exists():
                self.codes_with_prior.add(code)
        
        if self.use_cache:
            # Load pre-computed cache (supports both HDF5 and numpy formats)
            print(f"Loading pre-computed priors from {cache_path}...")
            cache_path_obj = Path(cache_path)
            
            if cache_path_obj.suffix == '.npz':
                # Load from numpy format
                print("  Detected numpy format (.npz)")
                npz_data = np.load(cache_path, allow_pickle=True)
                self.prior_matrix = npz_data['priors']
                self.cached_species = [s.decode('utf-8') if isinstance(s, bytes) else s 
                                      for s in npz_data['species_list']]
                self.cached_indices = npz_data['sample_indices']
                self.cache_file = None  # No HDF5 file handle needed
            else:
                # Load from HDF5 format
                print("  Detected HDF5 format (.h5)")
                import h5py
                self.cache_file = h5py.File(cache_path, 'r')
                self.prior_matrix = self.cache_file['priors']  # HDF5 dataset (memory-mapped)
                self.cached_species = [s.decode('utf-8') for s in self.cache_file['species_list']]
                self.cached_indices = self.cache_file['sample_indices'][:]
            
            # Create index mapping: df_index -> cache row
            if df_index is not None:
                self.index_map = {idx: i for i, idx in enumerate(self.cached_indices)}
            else:
                self.index_map = {i: i for i in range(len(self.cached_indices))}
            
            print(f"✓ Loaded cache: {len(self.cached_indices)} samples, {len(self.cached_species)} species")
            if hasattr(self.prior_matrix, 'nbytes'):
                print(f"  Cache size: {self.prior_matrix.nbytes / 1e6:.1f} MB")
            else:
                print(f"  Cache size: {self.prior_matrix.size * 4 / 1e6:.1f} MB (estimated)")
            
            # Verify species match
            if set(self.cached_species) != set(species_list):
                print(f"⚠️  Warning: Cache species don't match current species list")
                print(f"  Cache: {len(self.cached_species)} species")
                print(f"  Current: {len(species_list)} species")
        else:
            # Cache for prior queries (rounded coordinates + time bucket)
            self.cache = {}
            self.cache_hits = 0
            self.cache_misses = 0
            
            print(f"Prior wrapper: {len(self.codes_with_prior)}/{len(species_list)} species have prior files")
            print("  (Using on-the-fly computation - consider pre-computing cache for speed)")
    
    def _cache_key(self, lat: float, lon: float, season: int) -> tuple:
        """Create cache key from rounded coordinates and season."""
        # Round to 0.1 degree (~11km) for caching
        lat_rounded = round(lat * 10) / 10
        lon_rounded = round(lon * 10) / 10
        return (lat_rounded, lon_rounded, season)
    
    def get_prior_probs_batch(
        self,
        latitudes: list,
        longitudes: list,
        dates: list,
        sample_indices: list = None,
    ) -> np.ndarray:
        """
        Get prior probabilities with caching or from pre-computed cache.
        
        Args:
            latitudes: List of latitudes
            longitudes: List of longitudes
            dates: List of date strings
            sample_indices: Optional list of sample indices (for cache lookup)
        
        Returns:
            Array of shape (batch_size, num_species) with prior probabilities
        """
        batch_size = len(latitudes)
        num_species = len(self.species_list)
        
        if self.use_cache and sample_indices is not None:
            # Fast path: load from pre-computed cache
            prior_probs = np.zeros((batch_size, num_species), dtype=np.float32)
            
            for i, sample_idx in enumerate(sample_indices):
                if sample_idx in self.index_map:
                    cache_row = self.index_map[sample_idx]
                    # Get cached prior vector (handle both HDF5 and numpy formats)
                    cached_prior = self.prior_matrix[cache_row]
                    
                    # Convert to numpy array if needed (HDF5 datasets are already arrays)
                    if not isinstance(cached_prior, np.ndarray):
                        cached_prior = np.array(cached_prior)
                    
                    # Map to current species order if needed
                    if self.cached_species == self.species_list:
                        prior_probs[i] = cached_prior
                    else:
                        # Reorder to match current species list
                        species_to_idx = {s: j for j, s in enumerate(self.cached_species)}
                        for j, species in enumerate(self.species_list):
                            if species in species_to_idx:
                                prior_probs[i, j] = cached_prior[species_to_idx[species]]
                else:
                    
                    # raise Exception(f"Error: failed to find prior probability for i, sample_idx: {i, sample_idx}")
                    print(f"Warning: failed to find prior probability for i, sample_idx: {i, sample_idx}")
                    
                    # Fallback to small uniform value if index not found
                    # prior_probs[i] = np.ones(num_species) * 1e-6
            
            return prior_probs
        
        # Fallback to on-the-fly computation
        # Uniform prior as default
        print("Warning: assuming uniform prior.")
        uniform_prob = 1.0 / num_species
        prior_probs = np.ones((batch_size, num_species)) * uniform_prob
        
        # Convert dates to seasons (much faster - 4 seasons vs 52 weeks!)
        season_indices = []
        for date_str in dates:
            try:
                date = pd.to_datetime(date_str)
                week = date.isocalendar()[1]
                season = week_to_season(week)
                season_indices.append(season)
            except:
                print(f"Warning: failed to convert date_str: {date_str}.")
                season_indices.append(1)  # Default to Spring
        
        # Convert to arrays
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
            except:
                print(f"Warning: error for i, lat, lon: {i, lat, lon}")
                pass
        
        # Skip if no valid coordinates
        if not valid_mask.any():
            print("returning uniform dist of prior probs")
            return prior_probs
        
        # Group by season for batch lookups (only 4 seasons instead of 52 weeks!)
        unique_seasons = np.unique(season_indices)
        
        # Only process species with prior files (much faster!)
        valid_species_indices = [
            j for j, species in enumerate(self.species_list)
            if self.species_to_code.get(species) in self.codes_with_prior
        ]
        
        for season in unique_seasons:
            season_mask = (np.array(season_indices) == season) & valid_mask
            if not season_mask.any():
                continue
            
            season_coords = np.column_stack([lat_arr[season_mask], lon_arr[season_mask]])
            season_indices_local = np.where(season_mask)[0]
            
            # Batch lookup for each species using season-averaged abundance
            # This is much faster: 4 seasons vs 52 weeks = ~13x fewer queries!
            for j in valid_species_indices:
                species = self.species_list[j]
                ebird_code = self.species_to_code.get(species)
                
                try:
                    # Use season-averaged abundance (averages across all weeks in season)
                    # Use "point" method for speed - temporal averaging (13 weeks) compensates for lack of spatial averaging
                    probs = self.prior.probs_batch_season(ebird_code, season_coords, season=int(season), method="point", radius_km=5.0)
                    if len(probs.shape) > 1:
                        probs = probs.flatten()
                    
                    for local_i, global_i in enumerate(season_indices_local):
                        if local_i < len(probs) and probs[local_i] > 0:
                            prior_probs[global_i, j] = probs[local_i]
                except Exception:
                    raise Exception("Error parsing unique seasons.")
                    # Silently skip errors
                    pass
        
        # Return raw abundance scores (no normalization!)
        # Raw abundance scores are already informative (0.86 vs 0.01)
        # The model will learn how to scale/weight them via learnable parameters (w, temperature)
        # Just ensure no zeros (add small epsilon to avoid log(0))
        prior_probs = np.clip(prior_probs, 1e-8, 1.0)
        
        return prior_probs


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(
    model: WeightedFusionModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    device: str,
    prior_model: EBirdPriorWrapper,
    stage: str = "A",
    use_prior_in_training: bool = False,
):
    """Train for one epoch."""
    model.train()

    if stage == "A" or stage == "B":
        # Freeze audio encoder
        model.audio_encoder.eval()
        for param in model.audio_encoder.parameters():
            param.requires_grad = False
    
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Training (stage {stage})")
    import time
    batch_times = []
    encode_times = []
    
    for batch_idx, batch in enumerate(pbar):
        batch_start = time.time()
        audio = batch["audio"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        metadata = batch["metadata"]
        
        # Get prior probabilities
        if use_prior_in_training:
            latitudes = [m["latitude"] for m in metadata]
            longitudes = [m["longitude"] for m in metadata]
            dates = [m["date"] for m in metadata]
            sample_indices = batch.get("sample_indices", None)
            
            # Show progress for first batch (only if not using cache)
            if total == 0 and not prior_model.use_cache:
                print(f"\nComputing prior for first batch (this may take a minute to load cache)...")
                pbar.refresh()
            
            prior_probs = prior_model.get_prior_probs_batch(
                latitudes, longitudes, dates, sample_indices=sample_indices
            )
            prior_probs = torch.from_numpy(prior_probs).float().to(device)
            
            if total == 0 and not prior_model.use_cache:
                print("Prior computed, starting training...")
                pbar.refresh()
        else:
            # Use uniform prior during training for speed
            batch_size = len(metadata)
            num_classes = model.num_classes
            print(f"Using uniform prior distribution.")
            prior_probs = torch.ones(batch_size, num_classes, device=device) / num_classes
        
        optimizer.zero_grad()
        
        encode_start = time.time()
        with autocast("cuda", dtype=torch.bfloat16):
            final_logits, audio_logits, prior_robust = model(
                audio, prior_probs, metadata if model.use_gating else None
            )
            loss = F.cross_entropy(final_logits, labels)
            
            # TODO: this looks sketch
            # Regularization on weight
            if model.use_gating:
                # L2 regularization on gate outputs (computed in forward)
                reg_loss = 0.0  # Will add if needed
            else:
                reg_loss = 1e-3 * (model.w_weight ** 2)
            
            total_loss_batch = loss + reg_loss
        
        encode_time = time.time() - encode_start
        encode_times.append(encode_time)
        
        scaler.scale(total_loss_batch).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        total_loss += loss.item() * len(labels)
        preds = final_logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += len(labels)
        
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        
        # Show timing info every 10 batches
        if (batch_idx + 1) % 10 == 0:
            avg_batch_time = np.mean(batch_times[-10:])
            avg_encode_time = np.mean(encode_times[-10:])
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100*correct/total:.2f}%",
                "batch_t": f"{avg_batch_time:.2f}s",
                "encode_t": f"{avg_encode_time:.2f}s",
            })
        else:
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100*correct/total:.2f}%")
    
    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
    }


@torch.no_grad()
def evaluate(
    model: WeightedFusionModel,
    dataloader: DataLoader,
    device: str,
    prior_model: EBirdPriorWrapper,
    idx_to_label: dict,
    epoch: int,
    split: str = "val",
):
    """Evaluate model and create visualizations."""
    model.eval()
    
    all_final_logits = []
    all_audio_logits = []
    all_prior_probs = []
    all_labels = []
    all_metadata = []
    all_features = []  # For BEANS benchmark metrics
    total_loss = 0.0
    
    for batch in tqdm(dataloader, desc=f"Evaluating {split}"):
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
        
        # Convert labels to tensor if needed
        if isinstance(labels, torch.Tensor):
            labels_tensor = labels.to(device)
        else:
            labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)
        
        with autocast("cuda", dtype=torch.bfloat16):
            final_logits, audio_logits, prior_robust = model(
                audio, prior_probs_tensor, metadata if model.use_gating else None
            )
            # Compute loss for validation
            loss = F.cross_entropy(final_logits, labels_tensor)
            total_loss += loss.item() * len(labels_tensor)
            
            # Extract features for BEANS benchmark (from audio encoder)
            with torch.no_grad():
                features = model.audio_encoder(audio)  # Get encoder features
                all_features.append(features.float().cpu().detach())
        
        all_final_logits.append(final_logits.float().cpu().detach())
        all_audio_logits.append(audio_logits.float().cpu().detach())
        all_prior_probs.append(prior_robust.float().cpu().detach())
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
    
    # Average loss
    avg_loss = total_loss / len(labels)
    
    # Compute BEANS benchmark metrics (Probe, R-AUC, NMI)
    def compute_probe_accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
        """Probe accuracy = top-1 accuracy from logits."""
        preds = logits.argmax(axis=1)
        return accuracy_score(labels, preds)
    
    def compute_retrieval_auc(features: np.ndarray, labels: np.ndarray) -> float:
        """Compute retrieval ROC-AUC using cosine similarity."""
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
            return Exception(f"Failed to compute R-AUC")
    
    def compute_nmi(features: np.ndarray, labels: np.ndarray) -> float:
        """Compute Normalized Mutual Information using k-means clustering."""
        n_clusters = len(np.unique(labels))
        # Normalize features
        features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
        # K-means clustering
        kmeans = KMeans(n_clusters=min(n_clusters, len(features)), random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_norm)
        return normalized_mutual_info_score(labels, cluster_labels)
    
    # Compute BEANS metrics for Likelihood (audio) and Posterior (final)
    likelihood_probe = compute_probe_accuracy(audio_logits, labels)
    likelihood_rauc = compute_retrieval_auc(features, labels)
    likelihood_nmi = compute_nmi(features, labels)
    
    posterior_probe = compute_probe_accuracy(final_logits, labels)
    posterior_rauc = compute_retrieval_auc(features, labels)  # Same features, different logits
    posterior_nmi = compute_nmi(features, labels)  # Same features
    
    # Compute metrics
    final_preds = final_logits.argmax(axis=1)
    audio_preds = audio_logits.argmax(axis=1)
    prior_preds = prior_probs.argmax(axis=1)
    
    final_acc = accuracy_score(labels, final_preds)
    audio_acc = accuracy_score(labels, audio_preds)
    prior_acc = accuracy_score(labels, prior_preds)
    
    final_top5 = top_k_accuracy_score(labels, final_logits, k=5)
    audio_top5 = top_k_accuracy_score(labels, audio_logits, k=5)
    
    # Compute confusion matrix metrics (TP, FP, FN, TN) for each model
    num_classes = len(np.unique(labels))
    
    def compute_confusion_metrics(y_true, y_pred, num_classes):
        """Compute TP, FP, FN, TN from confusion matrix."""
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
        # For multi-class, we compute per-class and then aggregate
        # TP = diagonal, FP = row sum - TP, FN = col sum - TP, TN = total - TP - FP - FN
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        tn = cm.sum() - (tp + fp + fn)
        # Aggregate across all classes
        return {
            "tp": tp.sum(),
            "fp": fp.sum(),
            "fn": fn.sum(),
            "tn": tn.sum(),
            "fpr": fp.sum() / (fp.sum() + tn.sum()) if (fp.sum() + tn.sum()) > 0 else 0.0,
            "fnr": fn.sum() / (tp.sum() + fn.sum()) if (tp.sum() + fn.sum()) > 0 else 0.0,
        }
    
    final_cm = compute_confusion_metrics(labels, final_preds, num_classes)
    audio_cm = compute_confusion_metrics(labels, audio_preds, num_classes)
    prior_cm = compute_confusion_metrics(labels, prior_preds, num_classes)
    
    metrics = {
        "loss": avg_loss,  # Validation loss
        "final_acc": final_acc,
        "audio_acc": audio_acc,
        "prior_acc": prior_acc,
        "final_top5": final_top5,
        "audio_top5": audio_top5,
        # BEANS benchmark metrics - Likelihood (audio)
        "likelihood_probe": likelihood_probe,
        "likelihood_rauc": likelihood_rauc,
        "likelihood_nmi": likelihood_nmi,
        # BEANS benchmark metrics - Posterior (final)
        "posterior_probe": posterior_probe,
        "posterior_rauc": posterior_rauc,
        "posterior_nmi": posterior_nmi,
        # Confusion matrix metrics - Likelihood (audio)
        "likelihood_tp": audio_cm["tp"],
        "likelihood_fp": audio_cm["fp"],
        "likelihood_fn": audio_cm["fn"],
        "likelihood_tn": audio_cm["tn"],
        "likelihood_fpr": audio_cm["fpr"],
        "likelihood_fnr": audio_cm["fnr"],
        # Confusion matrix metrics - Posterior (final)
        "posterior_tp": final_cm["tp"],
        "posterior_fp": final_cm["fp"],
        "posterior_fn": final_cm["fn"],
        "posterior_tn": final_cm["tn"],
        "posterior_fpr": final_cm["fpr"],
        "posterior_fnr": final_cm["fnr"],
    }
    
    # Create top-10 comparison plots for individual samples
    if HAS_MATPLOTLIB:
        # Plot individual samples instead of aggregated
        sample_figs = plot_individual_samples(
            audio_logits, prior_probs, final_logits, labels, all_metadata, 
            idx_to_label, epoch, split, num_samples=10
        )
        if sample_figs and HAS_WANDB:
            for i, fig in enumerate(sample_figs):
                wandb.log({f"{split}/sample_{i}_top10": wandb.Image(fig)}, step=epoch)
                plt.close(fig)
    
    return metrics, {
        "final_logits": final_logits,
        "audio_logits": audio_logits,
        "prior_probs": prior_probs,
        "labels": labels,
        "metadata": all_metadata,
    }


def plot_individual_samples(
    audio_logits: np.ndarray,
    prior_probs: np.ndarray,
    final_logits: np.ndarray,
    labels: np.ndarray,
    metadata: list,
    idx_to_label: dict,
    epoch: int,
    split: str,
    num_samples: int = 10,
) -> list:
    """Plot top-10 species comparison for individual audio samples."""
    if not HAS_MATPLOTLIB:
        return []
    
    # Use fixed subset (random seed-stable)
    np.random.seed(42)
    if len(labels) > num_samples:
        indices = np.random.choice(len(labels), size=num_samples, replace=False)
    else:
        indices = np.arange(len(labels))
    
    figures = []
    
    for idx in indices:
        # Get single sample data
        audio_logit = audio_logits[idx:idx+1]
        prior_prob = prior_probs[idx:idx+1]
        final_logit = final_logits[idx:idx+1]
        true_label = labels[idx]
        meta = metadata[idx]
        
        # Convert to probabilities
        audio_prob = torch.softmax(torch.from_numpy(audio_logit), dim=1).numpy()[0]
        final_prob = torch.softmax(torch.from_numpy(final_logit), dim=1).numpy()[0]
        prior_prob_flat = prior_prob[0]
        
        # Normalize prior for visualization (only top-K to preserve informativeness)
        # For display purposes, normalize top 30 species
        K = 30
        topk_indices = np.argsort(prior_prob_flat)[-K:][::-1]
        prior_prob_normalized = prior_prob_flat.copy()
        topk_sum = prior_prob_flat[topk_indices].sum()
        if topk_sum > 1e-10:
            prior_prob_normalized[:] = 0
            prior_prob_normalized[topk_indices] = prior_prob_flat[topk_indices] / topk_sum
        
        # Get top-10 for each model
        top10_audio = np.argsort(audio_prob)[-10:][::-1]
        top10_prior = np.argsort(prior_prob_normalized)[-10:][::-1]
        top10_final = np.argsort(final_prob)[-10:][::-1]
        
        # Get all unique species in top-10s
        all_top = set(top10_audio) | set(top10_prior) | set(top10_final)
        all_top = sorted(list(all_top), key=lambda x: final_prob[x], reverse=True)[:10]
        
        species_names = [idx_to_label[i] for i in all_top]
        audio_vals = [audio_prob[i] for i in all_top]
        prior_vals = [prior_prob_normalized[i] for i in all_top]
        final_vals = [final_prob[i] for i in all_top]
        
        # Get metadata for header
        true_species = idx_to_label[true_label]
        date_str = meta.get("date", "Unknown") if meta else "Unknown"
        lat = meta.get("latitude", "?") if meta else "?"
        lon = meta.get("longitude", "?") if meta else "?"
        
        # Format coordinates
        try:
            lat_str = f"{float(lat):.2f}" if lat != "?" else "?"
            lon_str = f"{float(lon):.2f}" if lon != "?" else "?"
        except (ValueError, TypeError):
            lat_str = str(lat)
            lon_str = str(lon)
        
        # Create plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Header with metadata
        header = f"Sample {idx} | True: {true_species} | Date: {date_str} | Location: ({lat_str}, {lon_str})"
        fig.suptitle(header, fontsize=12, fontweight='bold')
        
        # Colors: highlight true label and top predictions
        colors_audio = []
        colors_prior = []
        colors_final = []
        for i, species_idx in enumerate(all_top):
            # Green if true label, blue if top-1 prediction, gray otherwise
            if species_idx == true_label:
                color = '#2ecc71'  # Green for true label
            elif i == 0:  # Top prediction
                color = '#3498db'  # Blue for top prediction
            else:
                color = '#95a5a6'  # Gray for others
            colors_audio.append(color)
            colors_prior.append(color)
            colors_final.append(color)
        
        axes[0].barh(range(len(species_names)), audio_vals, color=colors_audio, alpha=0.7)
        axes[0].set_yticks(range(len(species_names)))
        axes[0].set_yticklabels(species_names, fontsize=9)
        axes[0].set_xlabel('Probability', fontsize=10)
        axes[0].set_title('Audio Only\n(NatureLM)', fontsize=11, fontweight='bold')
        axes[0].invert_yaxis()
        axes[0].grid(axis='x', alpha=0.3)
        # Highlight true label
        if true_label in all_top:
            true_idx = all_top.index(true_label)
            axes[0].axhline(y=true_idx, color='#2ecc71', linestyle='--', linewidth=2, alpha=0.5)
        
        axes[1].barh(range(len(species_names)), prior_vals, color=colors_prior, alpha=0.7)
        axes[1].set_yticks(range(len(species_names)))
        axes[1].set_yticklabels(species_names, fontsize=9)
        axes[1].set_xlabel('Normalized Abundance', fontsize=10)
        axes[1].set_title('Space-Time Prior\n(AdaSTEM)', fontsize=11, fontweight='bold')
        axes[1].invert_yaxis()
        axes[1].grid(axis='x', alpha=0.3)
        if true_label in all_top:
            true_idx = all_top.index(true_label)
            axes[1].axhline(y=true_idx, color='#2ecc71', linestyle='--', linewidth=2, alpha=0.5)
        
        axes[2].barh(range(len(species_names)), final_vals, color=colors_final, alpha=0.7)
        axes[2].set_yticks(range(len(species_names)))
        axes[2].set_yticklabels(species_names, fontsize=9)
        axes[2].set_xlabel('Probability', fontsize=10)
        axes[2].set_title('Fused Model\n(Weighted)', fontsize=11, fontweight='bold')
        axes[2].invert_yaxis()
        axes[2].grid(axis='x', alpha=0.3)
        if true_label in all_top:
            true_idx = all_top.index(true_label)
            axes[2].axhline(y=true_idx, color='#2ecc71', linestyle='--', linewidth=2, alpha=0.5)
        
        plt.tight_layout()
        figures.append(fig)
    
    return figures


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train weighted fusion model")
    
    # Data
    parser.add_argument("--data_dir", type=str, default="../Data/cbi")
    parser.add_argument("--priors_dir", type=str, default="../Data/priors")
    parser.add_argument("--priors_cache", type=str, default=None,
                        help="Path to pre-computed HDF5 cache (use precompute_priors.py to create)")
    
    # Model
    parser.add_argument("--use_qformer", action="store_true")
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "max", "cls"])
    parser.add_argument("--use_gating", action="store_true", help="Use gating network (stage B) vs scalar (stage A)")
    parser.add_argument("--w_max", type=float, default=2.0, help="Maximum gate weight")
    parser.add_argument("--gate_hidden_dim", type=int, default=64)
    
    # Training
    parser.add_argument("--stage", type=str, default="A", choices=["A", "B", "C"],
                        help="A: scalar w, B: gating, C: fine-tune")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size (default: 64, can increase to 128+ if GPU memory allows)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--val_split", type=float, default=0.1)
    
    # Hardware
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    
    # Logging
    parser.add_argument("--wandb_project", type=str, default="weighted-fusion")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    
    # Checkpointing
    parser.add_argument("--save_dir", type=str, default="../checkpoints_fusion")
    parser.add_argument("--resume", type=str, default=None)
    
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true", help="Test setup only")
    
    return parser.parse_args()


def collate_fn(batch):
    """Custom collate to handle metadata and sample indices."""
    audio = torch.stack([b["audio"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch])
    metadata = [b["metadata"] for b in batch]
    sample_indices = [b.get("sample_idx", None) for b in batch]
    return {
        "audio": audio, 
        "label": labels, 
        "metadata": metadata,
        "sample_indices": sample_indices,
    }


def main():
    args = parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load data
    data_dir = Path(args.data_dir).resolve()
    train_csv = pd.read_csv(data_dir / "train.csv")

    found = []
    
    # iterate through train_csv and
    # remove any samples w/o valid audio paths
    for row in train_csv.iterrows():

        species_code = row[1]["ebird_code"]
        filename     = row[1]["filename"]
        # match the first '.ogg' or '.mp3' in audio_files
        audio_path = Path("/work/11295/leharris3/birdnoise/Data/cbi/train_audio") / species_code / filename
        # HACK: .mp3 -> .ogg
        audio_path = Path(audio_path.__str__().replace(".mp3", ".ogg"))
        found.append(audio_path.is_file())
    
    train_csv['found'] = found
    train_csv = train_csv[train_csv["found"] == True]
    print(f"HACK: remove missing .ogg files; # remaining: {len(train_csv)}")

    # Filter out BEANS test files (if using BEANS benchmark)
    # For now, use all CBI data
    print(f"Total CBI samples: {len(train_csv)}")
    
    # Create label mapping
    all_species = sorted(train_csv["species"].unique())
    label_to_idx = {s: i for i, s in enumerate(all_species)}
    idx_to_label = {i: s for s, i in label_to_idx.items()}
    num_classes = len(all_species)
    
    # Split data
    train_split, val_split = train_test_split(
        train_csv, test_size=args.val_split, stratify=train_csv["species"], random_state=args.seed
    )
    
    # Create datasets
    audio_dir = data_dir / "train_audio"
    train_dataset = CBIBirdDataset(audio_dir, train_split, label_to_idx)
    val_dataset = CBIBirdDataset(audio_dir, val_split, label_to_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    
    # Initialize model
    print("Loading NatureLM encoder (this may take a minute)...")
    audio_encoder = NatureLMAudioEncoder(use_qformer=args.use_qformer, pooling=args.pooling)
    model = WeightedFusionModel(
        audio_encoder=audio_encoder,
        num_classes=num_classes,
        use_gating=args.use_gating or args.stage in ["B", "C"],
        w_max=args.w_max,
        gate_hidden_dim=args.gate_hidden_dim,
    )
    model = model.to(device)
    
    # Print GPU memory usage
    if device == "cuda":
        print(f"GPU Memory - Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB, "
              f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
        print(f"Consider increasing --batch_size if you have GPU memory available (current: {args.batch_size})")
    
    # Initialize prior (with optional cache)
    cache_path = Path(args.priors_cache).resolve() if args.priors_cache else None
    prior_model = EBirdPriorWrapper(
        priors_dir=Path(args.priors_dir).resolve(),
        species_list=list(idx_to_label.values()),
        cbi_metadata_path=data_dir / "train.csv",
        cache_path=cache_path,
        df_index=train_csv.index,  # Pass original dataframe index for cache lookup
    )
    
    # Optimizer (only train fusion parameters in stage A/B)
    if args.stage in ["A", "B"]:
        # Freeze audio encoder
        for param in model.audio_encoder.parameters():
            param.requires_grad = False
        
        # Collect trainable parameters
        trainable_params = []
        trainable_params.extend(list(model.audio_classifier.parameters()))
        trainable_params.append(model.temperature)
        trainable_params.append(model.epsilon)
        
        if args.use_gating or args.stage == "B":
            trainable_params.extend(list(model.gate_network.parameters()))
        else:
            trainable_params.append(model.w_weight)
        
        print(f"Training {len(trainable_params)} parameter groups (stage {args.stage})")
    else:
        # Stage C: fine-tune everything
        trainable_params = list(model.parameters())
        print(f"Training all {sum(p.numel() for p in trainable_params):,} parameters (stage C)")
    
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    scaler = GradScaler("cuda")
    
    # Resume if specified
    start_epoch = 1
    best_val_acc = 0.0
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
        if "val_acc" in checkpoint:
            best_val_acc = checkpoint["val_acc"]
        print(f"Resumed from epoch {start_epoch}")
    
    # Scheduler (created after resume to set correct starting step)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    # TODO
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
    
    # Initialize scheduler at correct step if resuming
    last_epoch = (start_epoch - 1) * len(train_loader) if args.resume else -1
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
    
    # Wandb
    if HAS_WANDB and not args.no_wandb:
        run_name = args.wandb_run_name or f"fusion_{args.stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
    
    # Training loop
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save label mapping
    torch.save({"label_to_idx": label_to_idx, "idx_to_label": idx_to_label}, 
               save_dir / "label_mapping.pth")
    
    # Dry run check
    if args.dry_run:
        print("\n" + "="*60)
        print("DRY RUN - Testing one batch...")
        print("="*60)
        batch = next(iter(train_loader))
        audio = batch["audio"].to(device)
        metadata = batch["metadata"]
        latitudes = [m["latitude"] for m in metadata]
        longitudes = [m["longitude"] for m in metadata]
        dates = [m["date"] for m in metadata]
        sample_indices = batch.get("sample_indices", None)
        prior_probs = prior_model.get_prior_probs_batch(
            latitudes, longitudes, dates, sample_indices=sample_indices
        )
        prior_probs_tensor = torch.from_numpy(prior_probs).float().to(device)
        
        with torch.no_grad():
            final_logits, audio_logits, prior_robust = model(
                audio, prior_probs_tensor, metadata if model.use_gating else None
            )
        print(f"Final logits shape: {final_logits.shape}")
        print(f"Audio logits shape: {audio_logits.shape}")
        print(f"Prior robust shape: {prior_robust.shape}")
        print(f"Temperature: {model.temperature.item():.4f}")
        print(f"Epsilon: {model.epsilon.item():.4f}")
        if not model.use_gating:
            print(f"Weight w: {model.w_weight.item():.4f}")
        print("Dry run successful!")
        return
    
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs} (Stage {args.stage})")
        print("="*60)
        
        # Train (use real prior - model needs to see real priors to learn optimal fusion weights)
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, prior_model, args.stage,
            use_prior_in_training=True,  # Use real prior during training
        )
        
        # Validate
        val_metrics, viz_data = evaluate(
            model, val_loader, device, prior_model, idx_to_label, epoch, "val"
        )
        
        print(f"\nTrain Loss: {train_metrics['loss']:.4f}, Acc: {100*train_metrics['accuracy']:.2f}%")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val - Likelihood: {100*val_metrics['audio_acc']:.2f}%, "
              f"Prior: {100*val_metrics['prior_acc']:.2f}%, "
              f"Posterior: {100*val_metrics['final_acc']:.2f}%")
        print(f"\nBEANS Benchmark Metrics:")
        print(f"  Likelihood - Probe: {100*val_metrics['likelihood_probe']:.2f}%, "
              f"R-AUC: {val_metrics['likelihood_rauc']:.4f}, NMI: {val_metrics['likelihood_nmi']:.4f}")
        print(f"  Posterior  - Probe: {100*val_metrics['posterior_probe']:.2f}%, "
              f"R-AUC: {val_metrics['posterior_rauc']:.4f}, NMI: {val_metrics['posterior_nmi']:.4f}")
        print(f"\nConfusion Matrix Metrics:")
        print(f"  Likelihood - TP: {val_metrics['likelihood_tp']:4d}, FP: {val_metrics['likelihood_fp']:4d}, "
              f"FN: {val_metrics['likelihood_fn']:4d}, TN: {val_metrics['likelihood_tn']:4d}, "
              f"FPR: {100*val_metrics['likelihood_fpr']:.2f}%, FNR: {100*val_metrics['likelihood_fnr']:.2f}%")
        print(f"  Posterior  - TP: {val_metrics['posterior_tp']:4d}, FP: {val_metrics['posterior_fp']:4d}, "
              f"FN: {val_metrics['posterior_fn']:4d}, TN: {val_metrics['posterior_tn']:4d}, "
              f"FPR: {100*val_metrics['posterior_fpr']:.2f}%, FNR: {100*val_metrics['posterior_fnr']:.2f}%")
        
        # Log to wandb
        if HAS_WANDB and not args.no_wandb:
            log_dict = {
                "epoch": epoch,
                # Losses
                "train/loss": train_metrics["loss"],
                "val/loss": val_metrics["loss"],
                # Accuracies
                "train/acc": train_metrics["accuracy"],
                "val/audio_acc": val_metrics["audio_acc"],
                "val/prior_acc": val_metrics["prior_acc"],
                "val/final_acc": val_metrics["final_acc"],
                "val/final_top5": val_metrics["final_top5"],
                "val/audio_top5": val_metrics["audio_top5"],
                # BEANS benchmark - Likelihood (audio)
                "val/likelihood_probe": val_metrics["likelihood_probe"],
                "val/likelihood_rauc": val_metrics["likelihood_rauc"],
                "val/likelihood_nmi": val_metrics["likelihood_nmi"],
                # BEANS benchmark - Posterior (final)
                "val/posterior_probe": val_metrics["posterior_probe"],
                "val/posterior_rauc": val_metrics["posterior_rauc"],
                "val/posterior_nmi": val_metrics["posterior_nmi"],
                # Confusion matrix - Likelihood (audio)
                "val/likelihood_tp": val_metrics["likelihood_tp"],
                "val/likelihood_fp": val_metrics["likelihood_fp"],
                "val/likelihood_fn": val_metrics["likelihood_fn"],
                "val/likelihood_tn": val_metrics["likelihood_tn"],
                "val/likelihood_fpr": val_metrics["likelihood_fpr"],
                "val/likelihood_fnr": val_metrics["likelihood_fnr"],
                # Confusion matrix - Posterior (final)
                "val/posterior_tp": val_metrics["posterior_tp"],
                "val/posterior_fp": val_metrics["posterior_fp"],
                "val/posterior_fn": val_metrics["posterior_fn"],
                "val/posterior_tn": val_metrics["posterior_tn"],
                "val/posterior_fpr": val_metrics["posterior_fpr"],
                "val/posterior_fnr": val_metrics["posterior_fnr"],
                # Model parameters
                "model/temperature": model.temperature.item(),
                "model/epsilon": model.epsilon.item(),
            }
            
            if not model.use_gating:
                log_dict["model/w_weight"] = model.w_weight.item()
            
            wandb.log(log_dict, step=epoch)
        
        # Save best model
        if val_metrics["final_acc"] > best_val_acc:
            best_val_acc = val_metrics["final_acc"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": best_val_acc,
            }, save_dir / "best_model.pth")
            print(f"New best model saved! (acc: {100*best_val_acc:.2f}%)")
        
        # Save periodic checkpoint
        if epoch % 5 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_metrics["final_acc"],
            }, save_dir / f"checkpoint_epoch{epoch}.pth")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()