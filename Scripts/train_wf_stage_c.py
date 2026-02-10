"""
Stage C Training: Two-phase linear probe + fusion weight learning.

Phase 1: Train linear probe (classifier head) with frozen encoder
Phase 2: Freeze linear probe, train only the fusion weight "w"

Model: final_logits = audio_logits / T + w * log(prior + eps)
"""

import argparse
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

        sample = {
            "audio": torch.from_numpy(audio.astype(np.float32)),
            "label": self.label_to_idx[row["species"]],
            "metadata": {
                "latitude": float(row.get("latitude", 0.0) or 0.0),
                "longitude": float(row.get("longitude", 0.0) or 0.0),
                "day_of_year": day_of_year,
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

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
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


class StageCModel(nn.Module):
    """
    Simple fusion model for stage C training.

    final_logits = audio_logits / temperature + w * log(prior + epsilon)
    """

    def __init__(self, encoder: FrozenAudioEncoder, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes

        # Linear probe (classifier head)
        self.classifier = nn.Linear(encoder.encoder_dim, num_classes)

        # Learnable fusion parameters
        self.temperature = nn.Parameter(torch.ones(1))
        self.epsilon = nn.Parameter(torch.tensor(0.01))
        self.w = nn.Parameter(torch.tensor(0.5))

    def forward(self, audio: torch.Tensor, prior_probs: torch.Tensor):
        """
        Args:
            audio: (batch_size, audio_length)
            prior_probs: (batch_size, num_classes)

        Returns:
            final_logits: (batch_size, num_classes)
            audio_logits: (batch_size, num_classes)
        """
        # Encode audio and classify
        features = self.encoder(audio)
        audio_logits = self.classifier(features.float())

        # Temperature-scaled audio logits
        temp = self.temperature.abs().clamp(min=1e-8)
        audio_logits_scaled = audio_logits / temp

        # Robust prior with epsilon smoothing
        eps = self.epsilon.clamp(min=1e-8, max=0.1)
        log_prior = torch.log(prior_probs + eps)

        # Fused logits
        w = self.w.clamp(min=0.0, max=2.0)
        final_logits = audio_logits_scaled + w * log_prior

        return final_logits, audio_logits


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
                row_sums = np.where(row_sums > 0, row_sums, 1.0)  # Avoid division by zero
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
    model: StageCModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    prior_model: PriorWrapper,
    device: str,
    epochs: int,
    lr: float,
    use_wandb: bool = False,
):
    """
    Phase 1: Train linear probe with frozen encoder.
    Only the classifier head is trained.
    """
    print("\n" + "=" * 60)
    print("PHASE 1: Training Linear Probe")
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
                _, audio_logits = model(audio, prior_probs)
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

    print(f"\nPhase 1 complete. Best probe accuracy: {100*best_acc:.2f}%")
    return best_acc


def train_phase2(
    model: StageCModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    prior_model: PriorWrapper,
    device: str,
    epochs: int,
    lr: float,
    use_wandb: bool = False,
):
    """
    Phase 2: Freeze linear probe, train only the fusion weight "w".
    Also trains temperature and epsilon.
    """
    print("\n" + "=" * 60)
    print("PHASE 2: Training Fusion Weight")
    print("=" * 60)

    # Freeze classifier, train only fusion parameters
    for param in model.parameters():
        param.requires_grad = False
    model.temperature.requires_grad = True
    model.epsilon.requires_grad = True
    model.w.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable}")

    fusion_params = [model.temperature, model.epsilon, model.w]
    optimizer = torch.optim.AdamW(fusion_params, lr=lr, weight_decay=0.001)
    scaler = GradScaler("cuda")

    best_acc = 0.0
    best_w = model.w.item()
    best_temp = model.temperature.item()
    best_eps = model.epsilon.item()

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        model.encoder.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Phase 2 Epoch {epoch}/{epochs}")
        for batch in pbar:
            audio = batch["audio"].to(device)
            labels = batch["label"].to(device)
            sample_indices = batch["sample_indices"]

            prior_probs = prior_model.get_batch(sample_indices)
            prior_probs = torch.from_numpy(prior_probs).float().to(device)

            optimizer.zero_grad()

            with autocast("cuda", dtype=torch.bfloat16):
                final_logits, _ = model(audio, prior_probs)
                loss = F.cross_entropy(final_logits, labels)
                # Small regularization on w
                loss = loss + 1e-3 * (model.w ** 2)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * len(labels)
            preds = final_logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{100*correct/total:.2f}%",
                w=f"{model.w.item():.3f}"
            )

        train_loss = total_loss / total
        train_acc = correct / total

        # Validation
        val_metrics = evaluate(model, val_loader, prior_model, device, use_fusion=True)

        print(f"  Train Loss: {train_loss:.4f}, Acc: {100*train_acc:.2f}%")
        print(f"  Val   Loss: {val_metrics['loss']:.4f}, Acc: {100*val_metrics['accuracy']:.2f}%")
        print(f"  w={model.w.item():.4f}, T={model.temperature.item():.4f}, eps={model.epsilon.item():.6f}")

        if use_wandb and HAS_WANDB:
            wandb.log({
                "phase2/epoch": epoch,
                "phase2/train_loss": train_loss,
                "phase2/train_acc": train_acc,
                "phase2/val_loss": val_metrics['loss'],
                "phase2/val_acc": val_metrics['accuracy'],
                "phase2/w": model.w.item(),
                "phase2/temperature": model.temperature.item(),
                "phase2/epsilon": model.epsilon.item(),
            })

        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            best_w = model.w.item()
            best_temp = model.temperature.item()
            best_eps = model.epsilon.item()

    print(f"\nPhase 2 complete. Best fused accuracy: {100*best_acc:.2f}%")
    return best_acc, best_w, best_temp, best_eps


@torch.no_grad()
def evaluate(
    model: StageCModel,
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

    for batch in dataloader:
        audio = batch["audio"].to(device)
        labels = batch["label"].to(device)
        sample_indices = batch["sample_indices"]

        prior_probs = prior_model.get_batch(sample_indices)
        prior_probs = torch.from_numpy(prior_probs).float().to(device)

        with autocast("cuda", dtype=torch.bfloat16):
            final_logits, audio_logits = model(audio, prior_probs)
            logits = final_logits if use_fusion else audio_logits
            loss = F.cross_entropy(logits, labels)

        total_loss += loss.item() * len(labels)
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_logits.append(logits.float().cpu())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_logits = torch.cat(all_logits, dim=0).numpy()

    return {
        "loss": total_loss / len(all_labels),
        "accuracy": accuracy_score(all_labels, all_preds),
        "top5_accuracy": top_k_accuracy_score(all_labels, all_logits, k=5),
    }


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Stage C: Linear probe + fusion weight training")

    # Data
    parser.add_argument("--data_dir", type=str, default="../Data/cbi")
    parser.add_argument("--priors_cache", type=str, default=None)

    # Model
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "max", "cls"])

    # Training
    parser.add_argument("--phase1_epochs", type=int, default=10)
    parser.add_argument("--phase2_epochs", type=int, default=10)
    parser.add_argument("--phase1_lr", type=float, default=1e-3)
    parser.add_argument("--phase2_lr", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_split", type=float, default=0.1)

    # Hardware
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)

    # Logging
    parser.add_argument("--wandb_project", type=str, default="stage-c-training")
    parser.add_argument("--no_wandb", action="store_true")

    # Checkpointing
    parser.add_argument("--save_dir", type=str, default="../checkpoints_stage_c")

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

    # Initialize model
    encoder = FrozenAudioEncoder(pooling=args.pooling)
    model = StageCModel(encoder, num_classes).to(device)

    # Initialize prior model
    cache_path = Path(args.priors_cache).resolve() if args.priors_cache else None
    prior_model = PriorWrapper(list(idx_to_label.values()), cache_path)

    # Initialize wandb
    use_wandb = HAS_WANDB and not args.no_wandb
    if use_wandb:
        run_name = f"stage_c_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Train linear probe
    probe_acc = train_phase1(
        model, train_loader, val_loader, prior_model, device,
        epochs=args.phase1_epochs, lr=args.phase1_lr, use_wandb=use_wandb
    )

    # Save after phase 1
    torch.save({
        "model_state_dict": model.state_dict(),
        "probe_accuracy": probe_acc,
        "label_to_idx": label_to_idx,
    }, save_dir / "phase1_model.pth")

    # Phase 2: Train fusion weight
    final_acc, final_w, final_temp, final_eps = train_phase2(
        model, train_loader, val_loader, prior_model, device,
        epochs=args.phase2_epochs, lr=args.phase2_lr, use_wandb=use_wandb
    )

    # Save final model
    torch.save({
        "model_state_dict": model.state_dict(),
        "probe_accuracy": probe_acc,
        "final_accuracy": final_acc,
        "final_w": final_w,
        "final_temperature": final_temp,
        "final_epsilon": final_eps,
        "label_to_idx": label_to_idx,
    }, save_dir / "final_model.pth")

    # Print final results
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Phase 1 (Linear Probe) Accuracy: {100*probe_acc:.2f}%")
    print(f"Phase 2 (Fused Model)  Accuracy: {100*final_acc:.2f}%")
    print(f"\nFinal Learned Parameters:")
    print(f"  w (fusion weight): {final_w:.4f}")
    print(f"  temperature:       {final_temp:.4f}")
    print(f"  epsilon:           {final_eps:.6f}")
    print(f"\nModel saved to: {save_dir / 'final_model.pth'}")

    if use_wandb:
        wandb.log({
            "final/probe_accuracy": probe_acc,
            "final/fused_accuracy": final_acc,
            "final/w": final_w,
            "final/temperature": final_temp,
            "final/epsilon": final_eps,
        })
        wandb.finish()


if __name__ == "__main__":
    main()
