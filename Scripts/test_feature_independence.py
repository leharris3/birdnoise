"""
Test independence between audio embeddings and spatio-temporal priors.

Hypothesis: If audio embeddings e(x) and priors s are independent (low mutual information),
then a linear layer f trained to predict s_hat = f(e(x)) should not outperform
a baseline that always predicts the mean prior.

Experiment:
1. Extract audio embeddings e(x) from frozen NatureLM encoder
2. Load corresponding spatio-temporal priors s for each sample
3. Train a linear layer to predict s from e(x)
4. Compare MSE of predictions to mean-prior baseline

If MSE(f(e(x)), s) ≈ MSE(mean_prior, s), the features are independent.
If MSE(f(e(x)), s) << MSE(mean_prior, s), there is mutual information between them.
"""

import os
import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
import resampy

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from scipy import stats

warnings.filterwarnings("ignore", category=UserWarning)

# Add NatureLM to path
sys.path.insert(0, str(Path(__file__).parent.parent / "NatureLM-audio"))


# ============================================================================
# Dataset Classes (simplified from train_weighted_fusion.py)
# ============================================================================

class CBIBirdDatasetSimple(Dataset):
    """CBI dataset for feature independence testing."""

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
        self.max_length_seconds = max_length_seconds
        self.sample_rate = sample_rate
        self.max_samples = int(max_length_seconds * sample_rate)

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        row = self.metadata_df.iloc[idx]

        species_code = row["ebird_code"]
        filename = row["filename"]
        audio_path = self.audio_dir / species_code / filename
        audio_path = str(audio_path).replace(".mp3", ".ogg")

        try:
            audio, sr = sf.read(audio_path)
        except Exception:
            audio = np.zeros(self.max_samples, dtype=np.float32)
            sr = self.sample_rate

        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        if sr != self.sample_rate:
            audio = resampy.resample(audio, sr, self.sample_rate)

        if len(audio) > self.max_samples:
            audio = audio[:self.max_samples]
        elif len(audio) < self.max_samples:
            audio = np.pad(audio, (0, self.max_samples - len(audio)))

        label = self.label_to_idx[row["species"]]

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
            day_of_year = 182
            hour = 12

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

        original_idx = self.original_indices[idx]

        return {
            "audio": torch.from_numpy(audio.astype(np.float32)),
            "label": label,
            "metadata": metadata,
            "sample_idx": original_idx,
        }


# ============================================================================
# Audio Encoder (from train_weighted_fusion.py)
# ============================================================================

class NatureLMAudioEncoder(nn.Module):
    """Frozen NatureLM audio encoder."""

    def __init__(self, pooling: str = "mean"):
        super().__init__()
        self.pooling = pooling

        print("Loading NatureLM-audio...")
        from NatureLM.models import NatureLM
        self.naturelm = NatureLM.from_pretrained("EarthSpeciesProject/NatureLM-audio")

        for param in self.naturelm.parameters():
            param.requires_grad = False
        self.naturelm.eval()

        if self.naturelm.use_audio_Qformer or self.naturelm.max_pooling:
            self.encoder_dim = self.naturelm.llama_model.config.hidden_size
        else:
            self.encoder_dim = self.naturelm.beats.cfg.encoder_embed_dim

        print(f"Encoder dimension: {self.encoder_dim}")

    @torch.no_grad()
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract features from audio."""
        self.naturelm.eval()
        if audio.device != next(self.naturelm.parameters()).device:
            audio = audio.to(next(self.naturelm.parameters()).device)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            audio_embeds, audio_atts = self.naturelm.encode_audio(audio)

        if self.pooling == "mean":
            mask = audio_atts.unsqueeze(-1).float()
            features = (audio_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        elif self.pooling == "max":
            features = audio_embeds.max(dim=1)[0]
        else:
            features = audio_embeds[:, 0, :]

        return features.float()


# ============================================================================
# Prior Wrapper (simplified from train_weighted_fusion.py)
# ============================================================================

class EBirdPriorWrapper:
    """Wrapper for eBird prior with pre-computed cache."""

    def __init__(self, cache_path: Path, species_list: list):
        self.species_list = species_list

        print(f"Loading pre-computed priors from {cache_path}...")
        cache_path_obj = Path(cache_path)

        if cache_path_obj.suffix == '.npz':
            npz_data = np.load(cache_path, allow_pickle=True)
            self.prior_matrix = npz_data['priors']
            self.cached_species = [s.decode('utf-8') if isinstance(s, bytes) else s
                                  for s in npz_data['species_list']]
            self.cached_indices = npz_data['sample_indices']
        else:
            import h5py
            self.cache_file = h5py.File(cache_path, 'r')
            self.prior_matrix = self.cache_file['priors'][:]
            self.cached_species = [s.decode('utf-8') for s in self.cache_file['species_list']]
            self.cached_indices = self.cache_file['sample_indices'][:]

        self.index_map = {idx: i for i, idx in enumerate(self.cached_indices)}
        print(f"Loaded cache: {len(self.cached_indices)} samples, {len(self.cached_species)} species")

    def get_prior(self, sample_idx: int) -> np.ndarray:
        """Get prior for a single sample."""
        if sample_idx in self.index_map:
            cache_row = self.index_map[sample_idx]
            return self.prior_matrix[cache_row].astype(np.float32)
        else:
            return np.ones(len(self.species_list), dtype=np.float32) / len(self.species_list)


# ============================================================================
# Linear Predictor
# ============================================================================

class PriorPredictor(nn.Module):
    """Linear layer to predict prior from audio embedding."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# ============================================================================
# Main Experiment
# ============================================================================

def run_experiment(args):
    """Run the feature independence experiment."""

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load data
    data_dir = Path(args.data_dir).resolve()
    train_csv = pd.read_csv(data_dir / "train.csv")

    # Filter for valid audio files
    found = []
    for _, row in tqdm(train_csv.iterrows(), total=len(train_csv), desc="Checking audio files"):
        audio_path = Path(args.data_dir) / "train_audio" / row["ebird_code"] / row["filename"]
        audio_path = Path(str(audio_path).replace(".mp3", ".ogg"))
        found.append(audio_path.is_file())

    train_csv['found'] = found
    train_csv = train_csv[train_csv["found"] == True]
    print(f"Valid samples: {len(train_csv)}")

    # Create label mapping
    all_species = sorted(train_csv["species"].unique())
    label_to_idx = {s: i for i, s in enumerate(all_species)}
    idx_to_label = {i: s for s, i in label_to_idx.items()}
    num_classes = len(all_species)
    print(f"Number of classes: {num_classes}")

    # Sample subset
    if args.num_samples < len(train_csv):
        train_csv = train_csv.sample(n=args.num_samples, random_state=args.seed)
        print(f"Sampled {args.num_samples} samples for experiment")

    # Filter out classes with fewer than minimum required samples for stratified split
    # This ensures reliable per-class statistics and enables stratified splitting
    min_samples_per_class = args.min_samples_per_class
    class_counts = train_csv["species"].value_counts()
    valid_classes = class_counts[class_counts >= min_samples_per_class].index.tolist()
    excluded_classes = class_counts[class_counts < min_samples_per_class].index.tolist()

    samples_before = len(train_csv)
    train_csv = train_csv[train_csv["species"].isin(valid_classes)]
    samples_after = len(train_csv)

    print(f"\nClass filtering (min {min_samples_per_class} samples per class):")
    print(f"  Classes retained: {len(valid_classes)}")
    print(f"  Classes excluded: {len(excluded_classes)}")
    print(f"  Samples retained: {samples_after} ({100*samples_after/samples_before:.1f}%)")

    # Update label mapping after filtering
    all_species = sorted(train_csv["species"].unique())
    label_to_idx = {s: i for i, s in enumerate(all_species)}
    idx_to_label = {i: s for s, i in label_to_idx.items()}
    num_classes = len(all_species)
    print(f"  Final number of classes: {num_classes}")

    # Split into train/test using stratified split (now safe after filtering)
    train_split, test_split = train_test_split(
        train_csv, test_size=0.2, random_state=args.seed, stratify=train_csv["species"]
    )

    # Create datasets
    audio_dir = data_dir / "train_audio"
    train_dataset = CBIBirdDatasetSimple(audio_dir, train_split, label_to_idx)
    test_dataset = CBIBirdDatasetSimple(audio_dir, test_split, label_to_idx)

    def collate_fn(batch):
        return {
            "audio": torch.stack([b["audio"] for b in batch]),
            "label": torch.tensor([b["label"] for b in batch]),
            "metadata": [b["metadata"] for b in batch],
            "sample_idx": [b["sample_idx"] for b in batch],
        }

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn
    )

    # Load encoder
    encoder = NatureLMAudioEncoder(pooling="mean")
    encoder = encoder.to(device)
    encoder.eval()

    # Load prior wrapper
    prior_model = EBirdPriorWrapper(
        cache_path=Path(args.priors_cache).resolve(),
        species_list=list(idx_to_label.values()),
    )

    # Extract all embeddings and priors
    print("\nExtracting embeddings and priors...")

    def extract_features(loader, desc):
        embeddings_list = []
        priors_list = []
        labels_list = []

        for batch in tqdm(loader, desc=desc):
            audio = batch["audio"].to(device)
            sample_indices = batch["sample_idx"]
            labels = batch["label"]

            # Get audio embeddings
            with torch.no_grad():
                emb = encoder.encode(audio)
            embeddings_list.append(emb.cpu())

            # Get priors
            priors_batch = []
            for idx in sample_indices:
                priors_batch.append(prior_model.get_prior(idx))
            priors_list.append(torch.from_numpy(np.stack(priors_batch)))
            labels_list.append(labels)

        return (
            torch.cat(embeddings_list, dim=0),
            torch.cat(priors_list, dim=0),
            torch.cat(labels_list, dim=0),
        )

    train_embeddings, train_priors, train_labels = extract_features(train_loader, "Train set")
    test_embeddings, test_priors, test_labels = extract_features(test_loader, "Test set")

    print(f"\nTrain embeddings shape: {train_embeddings.shape}")
    print(f"Train priors shape: {train_priors.shape}")
    print(f"Test embeddings shape: {test_embeddings.shape}")
    print(f"Test priors shape: {test_priors.shape}")

    # Compute mean prior (baseline)
    mean_prior = train_priors.mean(dim=0)
    print(f"\nMean prior shape: {mean_prior.shape}")
    print(f"Mean prior sum: {mean_prior.sum():.4f}")
    print(f"Mean prior max: {mean_prior.max():.6f}, min: {mean_prior.min():.6f}")

    # Baseline MSE: always predict mean prior
    baseline_mse_train = F.mse_loss(
        mean_prior.unsqueeze(0).expand(len(train_priors), -1),
        train_priors
    ).item()
    baseline_mse_test = F.mse_loss(
        mean_prior.unsqueeze(0).expand(len(test_priors), -1),
        test_priors
    ).item()

    print(f"\nBaseline MSE (mean prior):")
    print(f"  Train: {baseline_mse_train:.6f}")
    print(f"  Test:  {baseline_mse_test:.6f}")

    # Train linear predictor
    print(f"\nTraining linear predictor s_hat = f(e(x))...")

    predictor = PriorPredictor(
        input_dim=train_embeddings.shape[1],
        output_dim=train_priors.shape[1]
    ).to(device)

    optimizer = torch.optim.Adam(predictor.parameters(), lr=args.lr)

    # Create tensor datasets for training
    train_emb_tensor = train_embeddings.to(device)
    train_prior_tensor = train_priors.to(device)
    test_emb_tensor = test_embeddings.to(device)
    test_prior_tensor = test_priors.to(device)

    best_test_mse = float('inf')
    train_losses = []
    test_losses = []

    for epoch in range(args.predictor_epochs):
        # Training
        predictor.train()

        # Mini-batch training
        indices = torch.randperm(len(train_emb_tensor))
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(indices), args.batch_size):
            batch_idx = indices[i:i+args.batch_size]
            emb_batch = train_emb_tensor[batch_idx]
            prior_batch = train_prior_tensor[batch_idx]

            optimizer.zero_grad()
            pred = predictor(emb_batch)
            loss = F.mse_loss(pred, prior_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        train_loss = epoch_loss / n_batches
        train_losses.append(train_loss)

        # Evaluation
        predictor.eval()
        with torch.no_grad():
            train_pred = predictor(train_emb_tensor)
            train_mse = F.mse_loss(train_pred, train_prior_tensor).item()

            test_pred = predictor(test_emb_tensor)
            test_mse = F.mse_loss(test_pred, test_prior_tensor).item()

        test_losses.append(test_mse)

        if test_mse < best_test_mse:
            best_test_mse = test_mse

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{args.predictor_epochs}: "
                  f"Train MSE={train_mse:.6f}, Test MSE={test_mse:.6f}")

    # ========================================================================
    # COMPREHENSIVE STATISTICAL ANALYSIS
    # ========================================================================

    print("\n")
    print("=" * 80)
    print("FEATURE INDEPENDENCE TEST: COMPREHENSIVE STATISTICAL ANALYSIS")
    print("=" * 80)

    # Get final predictions
    predictor.eval()
    with torch.no_grad():
        test_pred = predictor(test_emb_tensor)
        test_pred_np = test_pred.cpu().numpy()
        test_prior_np = test_prior_tensor.cpu().numpy()

    # ------------------------------------------------------------------------
    # SECTION 1: EXPERIMENTAL OVERVIEW (Non-expert summary)
    # ------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("SECTION 1: WHAT THIS EXPERIMENT TESTS")
    print("-" * 80)
    print("""
PURPOSE: Determine if audio content and location/time information are independent.

BACKGROUND: We have two sources of information about bird species:
  1. AUDIO EMBEDDINGS: Numerical representations of what the bird sounds like,
     extracted by a neural network (NatureLM) from audio recordings.
  2. SPATIO-TEMPORAL PRIORS: Probability estimates of which species are likely
     present based on WHERE and WHEN the recording was made (from eBird data).

HYPOTHESIS: If these two information sources are truly independent, then knowing
what a bird sounds like should NOT help us predict where/when it was recorded.

METHOD: We train a simple linear model to predict the location/time priors from
the audio embeddings. If the model can't do better than just guessing the average
prior, the features are independent. If it can, there's shared information.

WHY THIS MATTERS: Independent features are more valuable when combined because
each provides unique information. Correlated features may cause redundancy or
overfitting when used together in a classification model.
""")

    # ------------------------------------------------------------------------
    # SECTION 2: RAW METRICS
    # ------------------------------------------------------------------------
    print("-" * 80)
    print("SECTION 2: RAW PERFORMANCE METRICS")
    print("-" * 80)

    improvement = (baseline_mse_test - best_test_mse) / baseline_mse_test * 100

    print(f"""
BASELINE MODEL (always predicts the average prior):
  - Training set MSE: {baseline_mse_train:.6f}
  - Test set MSE:     {baseline_mse_test:.6f}

LEARNED LINEAR MODEL (predicts prior from audio embedding):
  - Best test set MSE:  {best_test_mse:.6f}
  - Final test set MSE: {test_losses[-1]:.6f}

RAW IMPROVEMENT: {improvement:.2f}% reduction in prediction error

INTERPRETATION: The linear model reduces prediction error by {improvement:.2f}%
compared to always guessing the average. A larger reduction suggests the audio
embeddings contain information about location/time (i.e., they are NOT independent).
""")

    # ------------------------------------------------------------------------
    # SECTION 3: STATISTICAL SIGNIFICANCE (Permutation Test)
    # ------------------------------------------------------------------------
    print("-" * 80)
    print("SECTION 3: STATISTICAL SIGNIFICANCE (Permutation Test)")
    print("-" * 80)
    print("""
WHAT IS A PERMUTATION TEST?
A permutation test checks if observed results could happen by chance. We randomly
shuffle the relationship between audio embeddings and priors many times, each time
measuring how well the model predicts. If our real model rarely beats these random
shuffles, the observed improvement is statistically significant.
""")

    n_permutations = args.n_permutations
    print(f"Running {n_permutations} permutations (this may take a moment)...")

    # Compute per-sample MSE for test set
    actual_per_sample_mse = ((test_pred_np - test_prior_np) ** 2).mean(axis=1)
    actual_mean_mse = actual_per_sample_mse.mean()

    # Baseline per-sample MSE
    mean_prior_np = mean_prior.cpu().numpy()
    baseline_per_sample_mse = ((mean_prior_np - test_prior_np) ** 2).mean(axis=1)

    # Permutation test: shuffle the mapping between embeddings and priors
    permuted_improvements = []
    for i in tqdm(range(n_permutations), desc="Permutation test"):
        perm_idx = np.random.permutation(len(test_prior_np))
        shuffled_priors = test_prior_np[perm_idx]
        perm_mse = ((test_pred_np - shuffled_priors) ** 2).mean()
        perm_baseline_mse = ((mean_prior_np - shuffled_priors) ** 2).mean()
        perm_improvement = (perm_baseline_mse - perm_mse) / perm_baseline_mse * 100
        permuted_improvements.append(perm_improvement)

    permuted_improvements = np.array(permuted_improvements)
    p_value = (permuted_improvements >= improvement).mean()

    print(f"""
PERMUTATION TEST RESULTS:
  - Observed improvement: {improvement:.2f}%
  - Mean permuted improvement: {permuted_improvements.mean():.2f}% +/- {permuted_improvements.std():.2f}%
  - Permuted improvement range: [{permuted_improvements.min():.2f}%, {permuted_improvements.max():.2f}%]
  - p-value: {p_value:.4f} (proportion of permutations >= observed)

INTERPRETATION:
""")
    if p_value < 0.001:
        print(f"  *** HIGHLY SIGNIFICANT (p < 0.001) ***")
        print(f"  The observed improvement is extremely unlikely to occur by chance.")
        print(f"  Strong evidence that audio embeddings contain location/time information.")
    elif p_value < 0.01:
        print(f"  ** SIGNIFICANT (p < 0.01) **")
        print(f"  The observed improvement is unlikely to occur by chance.")
        print(f"  Good evidence that audio embeddings contain location/time information.")
    elif p_value < 0.05:
        print(f"  * MARGINALLY SIGNIFICANT (p < 0.05) *")
        print(f"  The observed improvement is somewhat unlikely to occur by chance.")
        print(f"  Weak evidence of shared information between features.")
    else:
        print(f"  NOT SIGNIFICANT (p >= 0.05)")
        print(f"  The observed improvement could reasonably occur by chance.")
        print(f"  Insufficient evidence to reject the independence hypothesis.")

    # ------------------------------------------------------------------------
    # SECTION 4: EFFECT SIZE (Cohen's d)
    # ------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("SECTION 4: EFFECT SIZE (Cohen's d)")
    print("-" * 80)
    print("""
WHAT IS EFFECT SIZE?
While p-values tell us if an effect exists, effect size tells us how LARGE that
effect is. Cohen's d measures the difference in means relative to the variability
in the data. Guidelines: |d| < 0.2 = negligible, 0.2-0.5 = small, 0.5-0.8 = medium, > 0.8 = large.
""")

    # Cohen's d: (baseline_mse - predictor_mse) / pooled_std
    # Using per-sample MSE for proper variance estimation
    pooled_std = np.sqrt(
        (baseline_per_sample_mse.var() + actual_per_sample_mse.var()) / 2
    )
    cohens_d = (baseline_per_sample_mse.mean() - actual_per_sample_mse.mean()) / pooled_std

    print(f"""
EFFECT SIZE CALCULATION:
  - Baseline mean per-sample MSE: {baseline_per_sample_mse.mean():.6f}
  - Predictor mean per-sample MSE: {actual_per_sample_mse.mean():.6f}
  - Pooled standard deviation: {pooled_std:.6f}
  - Cohen's d: {cohens_d:.3f}

INTERPRETATION:
""")
    if abs(cohens_d) < 0.2:
        effect_label = "NEGLIGIBLE"
        print(f"  Effect size is {effect_label} (|d| < 0.2)")
        print(f"  The practical difference between models is trivially small.")
    elif abs(cohens_d) < 0.5:
        effect_label = "SMALL"
        print(f"  Effect size is {effect_label} (0.2 <= |d| < 0.5)")
        print(f"  There is a small but potentially meaningful difference.")
    elif abs(cohens_d) < 0.8:
        effect_label = "MEDIUM"
        print(f"  Effect size is {effect_label} (0.5 <= |d| < 0.8)")
        print(f"  The difference between models is practically meaningful.")
    else:
        effect_label = "LARGE"
        print(f"  Effect size is {effect_label} (|d| >= 0.8)")
        print(f"  The difference between models is substantial and practically important.")

    # ------------------------------------------------------------------------
    # SECTION 5: EXPLAINED VARIANCE (R-squared)
    # ------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("SECTION 5: EXPLAINED VARIANCE (R-squared)")
    print("-" * 80)
    print("""
WHAT IS R-SQUARED?
R-squared measures the proportion of variance in the priors that is explained by
the audio embeddings. R^2 = 0 means no explanatory power (complete independence),
R^2 = 1 means perfect prediction. It answers: "How much of the variation in
location/time information can be accounted for by the audio content?"
""")

    # Total variance in priors (relative to mean)
    ss_total = ((test_prior_np - mean_prior_np) ** 2).sum()
    # Residual variance (unexplained by model)
    ss_residual = ((test_prior_np - test_pred_np) ** 2).sum()
    # R-squared
    r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0

    # Adjusted R-squared (penalizes model complexity)
    n_samples_test = len(test_prior_np)
    n_features = test_prior_np.shape[1]
    n_params = train_embeddings.shape[1]  # input dimension
    adj_r_squared = 1 - (1 - r_squared) * (n_samples_test - 1) / (n_samples_test - n_params - 1)

    print(f"""
VARIANCE ANALYSIS:
  - Total sum of squares (SS_total): {ss_total:.4f}
  - Residual sum of squares (SS_residual): {ss_residual:.4f}
  - R-squared: {r_squared:.4f}
  - Adjusted R-squared: {adj_r_squared:.4f}
    (Adjusted R^2 penalizes for model complexity; negative values indicate overfitting)

INTERPRETATION:
  The audio embeddings explain {r_squared*100:.2f}% of the variance in the priors.
""")
    if r_squared < 0.01:
        print(f"  This is < 1%, indicating near-complete independence.")
    elif r_squared < 0.05:
        print(f"  This is < 5%, indicating very weak shared information.")
    elif r_squared < 0.15:
        print(f"  This is 5-15%, indicating weak-to-moderate shared information.")
    else:
        print(f"  This is >= 15%, indicating substantial shared information.")

    # ------------------------------------------------------------------------
    # SECTION 6: BOOTSTRAP CONFIDENCE INTERVALS
    # ------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("SECTION 6: BOOTSTRAP CONFIDENCE INTERVALS")
    print("-" * 80)
    print("""
WHAT ARE BOOTSTRAP CONFIDENCE INTERVALS?
Bootstrap sampling creates many "pseudo-datasets" by randomly resampling from our
data with replacement. For each pseudo-dataset, we compute the metric of interest.
The distribution of these values gives us a confidence interval - a range that
likely contains the true value with a specified probability (here, 95%).
""")

    n_bootstrap = args.n_bootstrap
    print(f"Running {n_bootstrap} bootstrap iterations...")

    bootstrap_improvements = []
    bootstrap_r_squared = []

    for i in tqdm(range(n_bootstrap), desc="Bootstrap"):
        # Resample with replacement
        boot_idx = np.random.choice(len(test_prior_np), size=len(test_prior_np), replace=True)
        boot_pred = test_pred_np[boot_idx]
        boot_prior = test_prior_np[boot_idx]

        # Compute metrics on bootstrap sample
        boot_baseline_mse = ((mean_prior_np - boot_prior) ** 2).mean()
        boot_pred_mse = ((boot_pred - boot_prior) ** 2).mean()
        boot_improvement = (boot_baseline_mse - boot_pred_mse) / boot_baseline_mse * 100
        bootstrap_improvements.append(boot_improvement)

        boot_ss_total = ((boot_prior - mean_prior_np) ** 2).sum()
        boot_ss_residual = ((boot_prior - boot_pred) ** 2).sum()
        boot_r2 = 1 - (boot_ss_residual / boot_ss_total) if boot_ss_total > 0 else 0
        bootstrap_r_squared.append(boot_r2)

    bootstrap_improvements = np.array(bootstrap_improvements)
    bootstrap_r_squared = np.array(bootstrap_r_squared)

    ci_improvement = np.percentile(bootstrap_improvements, [2.5, 97.5])
    ci_r_squared = np.percentile(bootstrap_r_squared, [2.5, 97.5])

    print(f"""
BOOTSTRAP CONFIDENCE INTERVALS (95%):

  Improvement over baseline:
    - Point estimate: {improvement:.2f}%
    - 95% CI: [{ci_improvement[0]:.2f}%, {ci_improvement[1]:.2f}%]

  R-squared:
    - Point estimate: {r_squared:.4f}
    - 95% CI: [{ci_r_squared[0]:.4f}, {ci_r_squared[1]:.4f}]

INTERPRETATION:
  We are 95% confident that the true improvement lies between {ci_improvement[0]:.2f}%
  and {ci_improvement[1]:.2f}%. If this interval includes zero, the improvement is
  not statistically reliable.
""")
    if ci_improvement[0] > 0:
        print(f"  The 95% CI excludes zero: the improvement IS statistically reliable.")
    else:
        print(f"  The 95% CI includes zero: the improvement is NOT statistically reliable.")

    # ------------------------------------------------------------------------
    # SECTION 7: CORRELATION ANALYSIS
    # ------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("SECTION 7: CORRELATION ANALYSIS")
    print("-" * 80)
    print("""
PEARSON CORRELATION measures linear relationship between predicted and actual
priors. Values range from -1 (perfect negative) to +1 (perfect positive).
Zero indicates no linear relationship.
""")

    # Overall correlation (flattened)
    pred_flat = test_pred_np.flatten()
    prior_flat = test_prior_np.flatten()
    correlation, corr_pvalue = stats.pearsonr(pred_flat, prior_flat)

    # Per-sample correlation
    per_sample_corr = []
    for i in range(len(test_pred_np)):
        if test_prior_np[i].std() > 1e-8 and test_pred_np[i].std() > 1e-8:
            corr, _ = stats.pearsonr(test_pred_np[i], test_prior_np[i])
            if not np.isnan(corr):
                per_sample_corr.append(corr)

    mean_per_sample_corr = np.mean(per_sample_corr) if per_sample_corr else 0
    std_per_sample_corr = np.std(per_sample_corr) if per_sample_corr else 0

    print(f"""
CORRELATION RESULTS:

  Overall (all prior elements, all samples):
    - Pearson r: {correlation:.4f}
    - p-value: {corr_pvalue:.2e}

  Per-sample (average correlation within each sample):
    - Mean r: {mean_per_sample_corr:.4f} +/- {std_per_sample_corr:.4f}
    - Number of valid samples: {len(per_sample_corr)}

INTERPRETATION:
""")
    if abs(correlation) < 0.1:
        print(f"  Overall correlation is negligible (|r| < 0.1).")
    elif abs(correlation) < 0.3:
        print(f"  Overall correlation is weak (0.1 <= |r| < 0.3).")
    elif abs(correlation) < 0.5:
        print(f"  Overall correlation is moderate (0.3 <= |r| < 0.5).")
    else:
        print(f"  Overall correlation is strong (|r| >= 0.5).")

    # ------------------------------------------------------------------------
    # SECTION 8: PER-CLASS STRATIFIED ANALYSIS
    # ------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("SECTION 8: PER-CLASS STRATIFIED ANALYSIS")
    print("-" * 80)
    print("""
STRATIFIED ANALYSIS examines whether results differ across bird species classes.
This is important because:
  1. Some species may be more geographically restricted than others
  2. Some species' calls may encode more environmental information
  3. Aggregated results can mask class-specific patterns
""")

    # Compute per-class metrics
    class_results = []
    train_class_counts = {}
    test_class_counts = {}
    for label in train_labels.numpy():
        train_class_counts[label] = train_class_counts.get(label, 0) + 1
    for label in test_labels.numpy():
        test_class_counts[label] = test_class_counts.get(label, 0) + 1

    for cls_idx in sorted(test_class_counts.keys()):
        cls_mask = (test_labels == cls_idx).numpy()
        n_test = cls_mask.sum()

        if n_test < 2:  # Need at least 2 samples for meaningful stats
            continue

        cls_priors = test_prior_np[cls_mask]
        cls_preds = test_pred_np[cls_mask]

        cls_baseline_mse = ((mean_prior_np - cls_priors) ** 2).mean()
        cls_pred_mse = ((cls_preds - cls_priors) ** 2).mean()
        cls_improvement = (cls_baseline_mse - cls_pred_mse) / cls_baseline_mse * 100 if cls_baseline_mse > 0 else 0

        # Per-class R-squared
        cls_ss_total = ((cls_priors - mean_prior_np) ** 2).sum()
        cls_ss_residual = ((cls_priors - cls_preds) ** 2).sum()
        cls_r2 = 1 - (cls_ss_residual / cls_ss_total) if cls_ss_total > 0 else 0

        species_name = idx_to_label[cls_idx]
        n_train = train_class_counts.get(cls_idx, 0)

        class_results.append({
            'class_idx': cls_idx,
            'species': species_name,
            'n_train': n_train,
            'n_test': n_test,
            'baseline_mse': cls_baseline_mse,
            'pred_mse': cls_pred_mse,
            'improvement': cls_improvement,
            'r_squared': cls_r2,
        })

    class_df = pd.DataFrame(class_results)

    # Summary statistics across classes
    print(f"""
AGGREGATE STATISTICS ACROSS {len(class_df)} CLASSES:

  Improvement (% reduction in MSE):
    - Mean:   {class_df['improvement'].mean():+.2f}%
    - Median: {class_df['improvement'].median():+.2f}%
    - Std:    {class_df['improvement'].std():.2f}%
    - Range:  [{class_df['improvement'].min():.2f}%, {class_df['improvement'].max():.2f}%]

  R-squared (variance explained):
    - Mean:   {class_df['r_squared'].mean():.4f}
    - Median: {class_df['r_squared'].median():.4f}
    - Std:    {class_df['r_squared'].std():.4f}
    - Range:  [{class_df['r_squared'].min():.4f}, {class_df['r_squared'].max():.4f}]

  Classes with positive improvement: {(class_df['improvement'] > 0).sum()} / {len(class_df)} ({100*(class_df['improvement'] > 0).mean():.1f}%)
  Classes with R^2 > 0.05: {(class_df['r_squared'] > 0.05).sum()} / {len(class_df)} ({100*(class_df['r_squared'] > 0.05).mean():.1f}%)
""")

    # Detailed breakdown for top classes
    print("\nDETAILED BREAKDOWN (Top 15 classes by test sample count):")
    print("-" * 100)
    print(f"{'Species':<35} {'N(train)':>8} {'N(test)':>8} {'Base MSE':>10} {'Pred MSE':>10} {'Improv.':>10} {'R^2':>8}")
    print("-" * 100)

    top_classes_df = class_df.nlargest(15, 'n_test')
    for _, row in top_classes_df.iterrows():
        print(f"{row['species'][:35]:<35} {row['n_train']:>8} {row['n_test']:>8} "
              f"{row['baseline_mse']:>10.6f} {row['pred_mse']:>10.6f} "
              f"{row['improvement']:>+9.1f}% {row['r_squared']:>8.4f}")

    print("-" * 100)

    # Classes with strongest/weakest effects
    print("\nCLASSES WITH STRONGEST POSITIVE EFFECT (Top 5 by improvement):")
    top_improvement = class_df.nlargest(5, 'improvement')
    for _, row in top_improvement.iterrows():
        print(f"  {row['species'][:40]:<40} Improvement: {row['improvement']:+.1f}%, R^2: {row['r_squared']:.4f}, N={row['n_test']}")

    print("\nCLASSES WITH WEAKEST/NEGATIVE EFFECT (Bottom 5 by improvement):")
    bottom_improvement = class_df.nsmallest(5, 'improvement')
    for _, row in bottom_improvement.iterrows():
        print(f"  {row['species'][:40]:<40} Improvement: {row['improvement']:+.1f}%, R^2: {row['r_squared']:.4f}, N={row['n_test']}")

    # ------------------------------------------------------------------------
    # SECTION 9: FINAL VERDICT
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SECTION 9: FINAL VERDICT")
    print("=" * 80)

    # Compute summary metrics for decision
    significant = p_value < 0.05
    ci_excludes_zero = ci_improvement[0] > 0
    effect_meaningful = abs(cohens_d) >= 0.2
    r2_meaningful = r_squared >= 0.01

    evidence_for_dependence = sum([significant, ci_excludes_zero, effect_meaningful, r2_meaningful])

    print(f"""
SUMMARY OF EVIDENCE:

  1. Statistical Significance:
     - p-value = {p_value:.4f}
     - {'[X] SIGNIFICANT' if significant else '[ ] Not significant'} (p < 0.05)

  2. Confidence Interval:
     - 95% CI for improvement: [{ci_improvement[0]:.2f}%, {ci_improvement[1]:.2f}%]
     - {'[X] EXCLUDES ZERO' if ci_excludes_zero else '[ ] Includes zero'} (reliable improvement)

  3. Effect Size:
     - Cohen's d = {cohens_d:.3f}
     - {'[X] MEANINGFUL' if effect_meaningful else '[ ] Negligible'} (|d| >= 0.2)

  4. Explained Variance:
     - R^2 = {r_squared:.4f}
     - {'[X] NON-TRIVIAL' if r2_meaningful else '[ ] Trivial'} (R^2 >= 0.01)

EVIDENCE SCORE: {evidence_for_dependence}/4 criteria met
""")

    if evidence_for_dependence == 0:
        verdict = "STRONG INDEPENDENCE"
        verdict_detail = """
The audio embeddings and spatio-temporal priors appear to be INDEPENDENT.
None of our statistical tests found meaningful evidence of shared information.
This is GOOD NEWS for multi-modal fusion: combining these features should
provide complementary information without redundancy."""
    elif evidence_for_dependence == 1:
        verdict = "LIKELY INDEPENDENT"
        verdict_detail = """
The audio embeddings and spatio-temporal priors appear to be MOSTLY INDEPENDENT.
Only one criterion suggests shared information, which may be due to chance.
These features can likely be combined effectively for multi-modal classification."""
    elif evidence_for_dependence == 2:
        verdict = "WEAK DEPENDENCE"
        verdict_detail = """
There is WEAK EVIDENCE of shared information between audio embeddings and priors.
Results are mixed - some tests suggest dependence, others do not. This level of
correlation is unlikely to significantly impact multi-modal fusion performance."""
    elif evidence_for_dependence == 3:
        verdict = "MODERATE DEPENDENCE"
        verdict_detail = """
There is MODERATE EVIDENCE of shared information between audio embeddings and priors.
Most tests suggest the features are not fully independent. Consider this when
designing fusion architectures - there may be some redundancy to account for."""
    else:
        verdict = "STRONG DEPENDENCE"
        verdict_detail = """
There is STRONG EVIDENCE of shared information between audio embeddings and priors.
All criteria indicate dependence. The audio embeddings may encode location/time
information, which could cause issues if combined naively with priors."""

    print(f"""
{'*' * 60}
VERDICT: {verdict}
{'*' * 60}
{verdict_detail}

KEY METRICS:
  - Improvement over baseline: {improvement:.2f}% [{ci_improvement[0]:.2f}%, {ci_improvement[1]:.2f}%]
  - Effect size (Cohen's d): {cohens_d:.3f} ({effect_label})
  - Variance explained (R^2): {r_squared:.4f}
  - p-value: {p_value:.4f}
""")

    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    return {
        "baseline_mse_train": baseline_mse_train,
        "baseline_mse_test": baseline_mse_test,
        "predictor_mse_best": best_test_mse,
        "predictor_mse_final": test_losses[-1],
        "improvement_pct": improvement,
        "improvement_ci_lower": ci_improvement[0],
        "improvement_ci_upper": ci_improvement[1],
        "p_value": p_value,
        "cohens_d": cohens_d,
        "effect_label": effect_label,
        "r_squared": r_squared,
        "r_squared_ci_lower": ci_r_squared[0],
        "r_squared_ci_upper": ci_r_squared[1],
        "correlation": correlation,
        "correlation_pvalue": corr_pvalue,
        "mean_per_sample_correlation": mean_per_sample_corr,
        "n_classes": len(class_df),
        "class_mean_improvement": class_df['improvement'].mean(),
        "class_median_improvement": class_df['improvement'].median(),
        "verdict": verdict,
        "evidence_score": evidence_for_dependence,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Test feature independence between audio and priors")

    parser.add_argument("--data_dir", type=str, default="../Data/cbi",
                        help="Path to CBI data directory")
    parser.add_argument("--priors_cache", type=str, required=True,
                        help="Path to pre-computed priors cache (.npz or .h5)")

    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of samples for experiment (default: 1000)")
    parser.add_argument("--min_samples_per_class", type=int, default=5,
                        help="Minimum samples per class for stratified analysis (default: 5)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for encoding")
    parser.add_argument("--predictor_epochs", type=int, default=10,
                        help="Number of epochs to train the linear predictor")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for predictor")
    parser.add_argument("--n_permutations", type=int, default=1000,
                        help="Number of permutations for significance testing (default: 1000)")
    parser.add_argument("--n_bootstrap", type=int, default=1000,
                        help="Number of bootstrap iterations for confidence intervals (default: 1000)")

    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of dataloader workers")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = run_experiment(args)
