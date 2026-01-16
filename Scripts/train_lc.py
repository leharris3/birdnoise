"""
Train a linear classifier on top of frozen NatureLM-audio encoder.

This script:
1. Loads the pre-trained NatureLM-audio model (BEATs encoder + Q-Former)
2. Freezes the encoder
3. Trains a linear classifier head on the CBI bird audio dataset
4. Logs training metrics to Weights & Biases (wandb)

Usage:
    python train_lc.py --data_dir ../Data/cbi --epochs 30 --batch_size 32
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, List

import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast  # Updated from torch.cuda.amp
from tqdm import tqdm
import soundfile as sf
import resampy

# Suppress MP3 decoding warnings (Xing stream size, etc.)
# These are harmless warnings from mpg123 about VBR MP3 files
warnings.filterwarnings("ignore", message=".*Xing stream size.*")
warnings.filterwarnings("ignore", message=".*Illegal Audio-MPEG-Header.*")

# Add NatureLM-audio to path
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "NatureLM-audio"))

import wandb
from sklearn.model_selection import train_test_split
from sklearn.metrics import top_k_accuracy_score


# ============================================================================
# Dataset
# ============================================================================

class CBIBirdDataset(Dataset):
    """Dataset for Cornell Bird Identification (CBI) challenge."""
    
    def __init__(
        self,
        df: pd.DataFrame,
        audio_dir: Path,
        label_to_idx: Dict[str, int],
        sample_rate: int = 16000,
        max_length_seconds: float = 10.0,
        augment: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.audio_dir = Path(audio_dir)
        self.label_to_idx = label_to_idx
        self.sample_rate = sample_rate
        self.max_length = int(sample_rate * max_length_seconds)
        self.augment = augment
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        
        # Load audio
        ebird_code = row["ebird_code"]
        filename = row["filename"]
        audio_path = self.audio_dir / ebird_code / filename
        
        try:
            audio, sr = sf.read(str(audio_path))
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # Return a silent audio on error
            audio = np.zeros(self.max_length, dtype=np.float32)
            sr = self.sample_rate
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Resample if necessary
        if sr != self.sample_rate:
            audio = resampy.resample(audio, sr, self.sample_rate)
        
        # Augmentation: random crop or pad
        if self.augment and len(audio) > self.max_length:
            # Random crop
            start = np.random.randint(0, len(audio) - self.max_length)
            audio = audio[start:start + self.max_length]
        else:
            # Take from beginning, pad if necessary
            if len(audio) > self.max_length:
                audio = audio[:self.max_length]
            elif len(audio) < self.max_length:
                pad_size = self.max_length - len(audio)
                audio = np.pad(audio, (0, pad_size), mode='constant')
        
        # Normalize to [-1, 1]
        audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
        
        # Get label
        label = self.label_to_idx[ebird_code]
        
        return {
            "audio": torch.from_numpy(audio),
            "label": torch.tensor(label, dtype=torch.long),
            "ebird_code": ebird_code,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader."""
    audios = torch.stack([b["audio"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    
    return {
        "audio": audios,
        "label": labels,
    }


# ============================================================================
# Model
# ============================================================================

class LinearClassifier(nn.Module):
    """Linear classifier head on top of frozen encoder."""
    
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(input_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, hidden_dim)
        x = self.dropout(x)
        return self.classifier(x)


class NatureLMLinearClassifier(nn.Module):
    """
    Full NatureLM-audio encoder (BEATs + Q-Former) with a linear classification head.
    The encoder is frozen; only the classifier head is trained.
    
    NOTE: Requires HuggingFace login and Llama access. Run:
        huggingface-cli login
    And accept the Llama license at:
        https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
    """
    
    def __init__(
        self,
        num_classes: int,
        dropout: float = 0.1,
        device: str = "cuda",
        pooling: str = "mean",  # "mean", "max", or "cls"
        use_qformer: bool = True,  # If True, use BEATs + Q-Former; if False, use BEATs only
    ):
        super().__init__()
        self.device = device
        self.pooling = pooling
        self.use_qformer = use_qformer
        
        # Load pre-trained NatureLM-audio model
        print("Loading NatureLM-audio from HuggingFace...")
        print("(This requires HuggingFace login and Llama license acceptance)")
        from NatureLM.models import NatureLM
        
        self.naturelm = NatureLM.from_pretrained("EarthSpeciesProject/NatureLM-audio")
        self.naturelm = self.naturelm.to(device)
        
        # Freeze the entire encoder
        self._freeze_encoder()
        
        # Determine encoder output dimension
        if use_qformer:
            # Q-Former output is projected to LLM hidden size via audio_llama_proj
            self.encoder_dim = self.naturelm.llama_model.config.hidden_size  # 4096 for Llama 3.1 8B
        else:
            # Direct BEATs output dimension
            self.encoder_dim = self.naturelm.beats.cfg.encoder_embed_dim  # 768
        
        # Classification head
        self.classifier = LinearClassifier(self.encoder_dim, num_classes, dropout)
        
        print(f"Encoder dim: {self.encoder_dim}, Num classes: {num_classes}")
        print(f"Using Q-Former: {use_qformer}")
        
    def _freeze_encoder(self):
        """Freeze all encoder parameters."""
        # Freeze BEATs
        for param in self.naturelm.beats.parameters():
            param.requires_grad = False
        self.naturelm.beats.eval()
        
        # Freeze Q-Former
        if hasattr(self.naturelm, 'audio_Qformer'):
            for param in self.naturelm.audio_Qformer.parameters():
                param.requires_grad = False
            self.naturelm.audio_Qformer.eval()
            
        # Freeze layer norm
        if hasattr(self.naturelm, 'ln_audio'):
            for param in self.naturelm.ln_audio.parameters():
                param.requires_grad = False
                
        # Freeze audio query tokens
        if hasattr(self.naturelm, 'audio_query_tokens'):
            self.naturelm.audio_query_tokens.requires_grad = False
        
        # Freeze LLM components (we don't use them, but freeze anyway)
        if hasattr(self.naturelm, 'llama_model'):
            for param in self.naturelm.llama_model.parameters():
                param.requires_grad = False
                
        if hasattr(self.naturelm, 'audio_llama_proj'):
            for param in self.naturelm.audio_llama_proj.parameters():
                param.requires_grad = False
            
        print("Encoder frozen (BEATs + Q-Former + LLM).")
        
    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings from audio using NatureLM encoder.
        
        Args:
            audio: (batch, samples) raw waveform at 16kHz
            
        Returns:
            embeddings: (batch, hidden_dim) pooled features
        """
        with torch.no_grad():
            with torch.autocast(self.device, dtype=torch.bfloat16):
                if self.use_qformer:
                    # Use full NatureLM encode_audio (BEATs + Q-Former)
                    # Returns (audio_embeds, audio_atts)
                    audio_embeds, audio_atts = self.naturelm.encode_audio(audio, audio_padding_mask=None)
                    # audio_embeds shape: (batch, num_query_tokens * num_windows, hidden_dim)
                    
                    # Pool across the sequence dimension
                    if self.pooling == "mean":
                        embeddings = audio_embeds.mean(dim=1)
                    elif self.pooling == "max":
                        embeddings = audio_embeds.max(dim=1)[0]
                    elif self.pooling == "cls":
                        embeddings = audio_embeds[:, 0]
                    else:
                        raise ValueError(f"Unknown pooling: {self.pooling}")
                else:
                    # Use BEATs only
                    features, padding_mask = self.naturelm.beats(audio, padding_mask=None)
                    
                    if self.pooling == "mean":
                        if padding_mask is not None:
                            mask = ~padding_mask
                            features = features * mask.unsqueeze(-1).float()
                            embeddings = features.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
                        else:
                            embeddings = features.mean(dim=1)
                    elif self.pooling == "max":
                        embeddings = features.max(dim=1)[0]
                    elif self.pooling == "cls":
                        embeddings = features[:, 0]
                    else:
                        raise ValueError(f"Unknown pooling: {self.pooling}")
                
        return embeddings.float()  # Convert back to fp32 for classifier
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode audio and classify.
        
        Args:
            audio: (batch, samples) raw waveform at 16kHz
            
        Returns:
            logits: (batch, num_classes)
        """
        embeddings = self.encode_audio(audio)
        logits = self.classifier(embeddings)
        return logits
    
    def train(self, mode: bool = True):
        """Override train to keep encoder in eval mode."""
        super().train(mode)
        # Always keep encoder in eval mode
        self.naturelm.beats.eval()
        if hasattr(self.naturelm, 'audio_Qformer'):
            self.naturelm.audio_Qformer.eval()
        return self


# ============================================================================
# Training
# ============================================================================

def train_epoch(
    model: NatureLMLinearClassifier,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: GradScaler,
    device: str,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, batch in enumerate(pbar):
        audio = batch["audio"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision forward
        with autocast('cuda', dtype=torch.bfloat16):
            logits = model(audio)
            loss = F.cross_entropy(logits, labels)
        
        # Backward with gradient scaling
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        if scheduler is not None:
            scheduler.step()
        
        # Metrics
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100 * correct / total:.2f}%",
        })
    
    return {
        "train_loss": total_loss / len(train_loader),
        "train_acc": correct / total,
    }


@torch.no_grad()
def evaluate(
    model: NatureLMLinearClassifier,
    val_loader: DataLoader,
    device: str,
    epoch: int,
) -> Dict[str, float]:
    """Evaluate the model."""
    model.eval()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_logits = []
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
    
    for batch in pbar:
        audio = batch["audio"].to(device)
        labels = batch["label"].to(device)
        
        with autocast('cuda', dtype=torch.bfloat16):
            logits = model(audio)
            loss = F.cross_entropy(logits, labels)
        
        total_loss += loss.item()
        
        all_logits.append(logits.float().cpu())  # Convert BFloat16 to Float32
        all_preds.append(logits.argmax(dim=-1).cpu())
        all_labels.append(labels.cpu())
    
    # Concatenate and convert to numpy (ensure float32 for numpy compatibility)
    all_logits = torch.cat(all_logits, dim=0).float().numpy()
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    # Metrics
    acc = (all_preds == all_labels).mean()
    
    # Top-k accuracy
    top3_acc = top_k_accuracy_score(all_labels, all_logits, k=3, labels=range(all_logits.shape[1]))
    top5_acc = top_k_accuracy_score(all_labels, all_logits, k=5, labels=range(all_logits.shape[1]))
    
    return {
        "val_loss": total_loss / len(val_loader),
        "val_acc": acc,
        "val_top3_acc": top3_acc,
        "val_top5_acc": top5_acc,
    }


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train linear classifier on NatureLM-audio")
    
    # Data
    parser.add_argument("--data_dir", type=str, default="../Data/cbi",
                        help="Path to CBI dataset directory")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Validation split ratio")
    
    # Training
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--warmup_epochs", type=int, default=2,
                        help="Number of warmup epochs")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    
    # Model
    parser.add_argument("--pooling", type=str, default="mean",
                        choices=["mean", "max", "cls"],
                        help="Pooling strategy for encoder output")
    parser.add_argument("--max_length_seconds", type=float, default=10.0,
                        help="Maximum audio length in seconds")
    parser.add_argument("--use_qformer", action="store_true",
                        help="Use full NatureLM encoder (BEATs + Q-Former). If not set, uses BEATs only.")
    
    # System
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loader workers")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # Wandb
    parser.add_argument("--wandb_project", type=str, default="bird-classifier",
                        help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="W&B run name (auto-generated if not provided)")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable wandb logging")
    
    # Checkpointing
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--save_every", type=int, default=5,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    
    # Debug/Testing
    parser.add_argument("--dry_run", action="store_true",
                        help="Test setup only: load model and data, then exit without training")
    
    return parser.parse_args()


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
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Data paths
    data_dir = Path(args.data_dir).resolve()
    train_csv = data_dir / "train.csv"
    audio_dir = data_dir / "train_audio"
    
    print(f"Data directory: {data_dir}")
    print(f"Train CSV: {train_csv}")
    print(f"Audio directory: {audio_dir}")
    
    # Load data
    df = pd.read_csv(train_csv)
    print(f"Total samples: {len(df)}")
    print(f"Unique species: {df['ebird_code'].nunique()}")
    
    # Create label mapping
    unique_labels = sorted(df["ebird_code"].unique())
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    num_classes = len(unique_labels)
    print(f"Number of classes: {num_classes}")
    
    # Train/val split (stratified by species)
    train_df, val_df = train_test_split(
        df, 
        test_size=args.val_split, 
        stratify=df["ebird_code"],
        random_state=args.seed
    )
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    
    # Datasets
    train_dataset = CBIBirdDataset(
        train_df, audio_dir, label_to_idx,
        max_length_seconds=args.max_length_seconds,
        augment=True,
    )
    val_dataset = CBIBirdDataset(
        val_df, audio_dir, label_to_idx,
        max_length_seconds=args.max_length_seconds,
        augment=False,
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    # Model
    model = NatureLMLinearClassifier(
        num_classes=num_classes,
        dropout=args.dropout,
        device=device,
        pooling=args.pooling,
        use_qformer=args.use_qformer,
    )
    model = model.to(device)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
          f"({100*trainable_params/total_params:.2f}%)")
    
    # Dry run: just test loading, then exit
    if args.dry_run:
        print("\n" + "="*60)
        print("DRY RUN: Testing one forward pass...")
        print("="*60)
        
        # Test with one batch
        test_batch = next(iter(train_loader))
        test_audio = test_batch["audio"].to(device)
        
        with torch.no_grad():
            with torch.autocast(device, dtype=torch.bfloat16):
                test_output = model(test_audio)
        
        print(f"Input shape: {test_audio.shape}")
        print(f"Output shape: {test_output.shape}")
        print(f"Output sample: {test_output[0, :5]}")
        print("\n✓ Dry run successful! Model and data loading works.")
        print("  Remove --dry_run flag to start training.")
        return
    
    # Optimizer (only for classifier head)
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Scheduler: linear warmup then cosine decay
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Mixed precision scaler
    scaler = GradScaler('cuda')
    
    # Resume from checkpoint if specified
    start_epoch = 1
    best_val_acc = 0.0
    
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            print(f"\nResuming from checkpoint: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device)
            
            # Load model weights
            model.classifier.load_state_dict(checkpoint["model_state_dict"])
            print(f"  Loaded model weights")
            
            # Load optimizer state if available
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                print(f"  Loaded optimizer state")
            
            # Load scheduler state if available
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                print(f"  Loaded scheduler state")
            
            # Get starting epoch (resume from next epoch)
            if "epoch" in checkpoint:
                start_epoch = checkpoint["epoch"] + 1
                print(f"  Resuming from epoch {start_epoch}")
            
            # Get best val accuracy
            if "val_acc" in checkpoint:
                best_val_acc = checkpoint["val_acc"]
                print(f"  Previous best val acc: {100*best_val_acc:.2f}%")
            elif "best_val_acc" in checkpoint:
                best_val_acc = checkpoint["best_val_acc"]
                print(f"  Previous best val acc: {100*best_val_acc:.2f}%")
            
            print(f"  Resume complete!\n")
        else:
            print(f"Warning: Checkpoint not found at {resume_path}, starting from scratch")
    
    # Initialize wandb
    if not args.no_wandb:
        run_name = args.wandb_run_name or f"lc_{args.pooling}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
            resume="allow" if args.resume else None,
        )
        wandb.watch(model.classifier, log="all", log_freq=100)
    
    # Save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save label mapping
    label_map_path = save_dir / "label_mapping.pth"
    torch.save({
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
    }, label_map_path)
    print(f"Saved label mapping to {label_map_path}")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, epoch
        )
        
        # Validate
        val_metrics = evaluate(model, val_loader, device, epoch)
        
        # Log metrics
        metrics = {**train_metrics, **val_metrics, "epoch": epoch, "lr": scheduler.get_last_lr()[0]}
        
        print(f"\nTrain Loss: {train_metrics['train_loss']:.4f}, "
              f"Train Acc: {100*train_metrics['train_acc']:.2f}%")
        print(f"Val Loss: {val_metrics['val_loss']:.4f}, "
              f"Val Acc: {100*val_metrics['val_acc']:.2f}%, "
              f"Top-3: {100*val_metrics['val_top3_acc']:.2f}%, "
              f"Top-5: {100*val_metrics['val_top5_acc']:.2f}%")
        
        if not args.no_wandb:
            wandb.log(metrics)
        
        # Save best model
        if val_metrics["val_acc"] > best_val_acc:
            best_val_acc = val_metrics["val_acc"]
            best_path = save_dir / "best_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.classifier.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": best_val_acc,
                "config": vars(args),
            }, best_path)
            print(f"✓ New best model saved to {best_path} (val_acc: {100*best_val_acc:.2f}%)")
        
        # Save periodic checkpoint
        if epoch % args.save_every == 0:
            ckpt_path = save_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.classifier.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_acc": val_metrics["val_acc"],
                "config": vars(args),
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")
    
    # Final save
    final_path = save_dir / "final_model.pth"
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.classifier.state_dict(),
        "val_acc": val_metrics["val_acc"],
        "best_val_acc": best_val_acc,
        "config": vars(args),
    }, final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")
    print(f"Best validation accuracy: {100*best_val_acc:.2f}%")
    
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()

