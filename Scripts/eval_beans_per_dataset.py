"""
Evaluate models on BEANS benchmark datasets.

Computes:
1. Accuracy on classification datasets: esc50, watkins, bats, cbi, dogs, humbugdb, speech
2. mAP on detection datasets: dcase, enabirds, hiceas, rfcx, hainan, gibbons

Reports results with Mean column for each metric type.

Reference: https://github.com/earthspecies/beans
"""

import os
import sys
from pathlib import Path

# ============================================================================
# HuggingFace Cache Configuration - MUST be set BEFORE importing HF libraries
# ============================================================================
# This prevents HuggingFace from using ~/.cache/huggingface as fallback

# Define project-local cache directory
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
_HF_CACHE_DIR = _PROJECT_ROOT / ".cache" / "huggingface"


def _setup_hf_cache_environment():
    """
    Configure all HuggingFace-related environment variables to use project-local cache.

    This function MUST be called before importing any HuggingFace libraries (datasets,
    transformers, huggingface_hub) to ensure they don't use the home directory.

    Environment variables set:
    - HF_HOME: Main HuggingFace home directory
    - HF_DATASETS_CACHE: Cache for datasets library
    - HUGGINGFACE_HUB_CACHE: Cache for huggingface_hub downloads
    - TRANSFORMERS_CACHE: Cache for transformers models (legacy)
    - HF_HUB_CACHE: Alternative hub cache variable
    - HF_DATASETS_OFFLINE: Set to 0 (allow downloads, but to correct location)
    """
    cache_dir = str(_HF_CACHE_DIR)

    # Create cache directory if it doesn't exist
    _HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Set all relevant HuggingFace environment variables
    hf_env_vars = {
        "HF_HOME": cache_dir,
        "HF_DATASETS_CACHE": str(_HF_CACHE_DIR / "datasets"),
        "HUGGINGFACE_HUB_CACHE": str(_HF_CACHE_DIR / "hub"),
        "HF_HUB_CACHE": str(_HF_CACHE_DIR / "hub"),
        "TRANSFORMERS_CACHE": str(_HF_CACHE_DIR / "transformers"),
    }

    for var, value in hf_env_vars.items():
        os.environ[var] = value


def _validate_cache_not_in_homedir():
    """
    Validate that HuggingFace cache is NOT configured to use the home directory.

    Raises:
        RuntimeError: If any HF cache variable points to home directory.
    """
    home_dir = Path.home()

    # Check environment variables
    hf_vars_to_check = [
        "HF_HOME",
        "HF_DATASETS_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "HF_HUB_CACHE",
        "TRANSFORMERS_CACHE",
    ]

    violations = []
    for var in hf_vars_to_check:
        value = os.environ.get(var, "")
        if value:
            var_path = Path(value).resolve()
            try:
                # Check if the path is relative to home directory
                var_path.relative_to(home_dir)
                violations.append(f"  {var}={value}")
            except ValueError:
                # Not relative to home - this is good
                pass

    if violations:
        raise RuntimeError(
            f"HuggingFace cache variables point to home directory!\n"
            f"Violations:\n" + "\n".join(violations) + "\n"
            f"Please ensure all HF cache is configured to use project directory."
        )

    # Also check that our configured cache is not in home
    try:
        _HF_CACHE_DIR.resolve().relative_to(home_dir)
        raise RuntimeError(
            f"Project HF_CACHE_DIR is inside home directory: {_HF_CACHE_DIR}\n"
            f"Please configure a cache directory outside of {home_dir}"
        )
    except ValueError:
        # Not relative to home - this is good
        pass


# Setup cache environment BEFORE any HuggingFace imports
_setup_hf_cache_environment()
_validate_cache_not_in_homedir()

# ============================================================================
# Standard imports (AFTER cache configuration)
# ============================================================================

import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score
)

warnings.filterwarnings("ignore")

# Add NatureLM to path
sys.path.insert(0, str(Path(__file__).parent.parent / "NatureLM-audio"))

try:
    from datasets import load_dataset
    HAS_HF_DATASETS = True
except ImportError:
    HAS_HF_DATASETS = False
    print("Warning: huggingface datasets not installed. Run: pip install datasets")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed, visualizations disabled")

# Import model classes from train_weighted_fusion
from train_weighted_fusion import NatureLMAudioEncoder


# ============================================================================
# Configuration
# ============================================================================

# Cache directory for downloaded datasets (uses pre-configured path from setup)
HF_CACHE_DIR = _HF_CACHE_DIR

# Classification datasets (use accuracy)
ACCURACY_DATASETS = ["esc50", "watkins", "bats", "cbi", "dogs", "humbugdb", "speech"]

# Detection datasets (use mAP)
MAP_DATASETS = ["dcase", "enabirds", "hiceas", "rfcx", "hainan", "gibbons"]

# Dataset name mapping (BEANS-Zero uses slightly different names)
DATASET_NAME_MAP = {
    "esc50": "esc-50",
    "watkins": "watkins",
    "bats": "bats",
    "cbi": "cbi",
    "dogs": "dogs",
    "humbugdb": "humbugdb",
    "speech": "speech_commands",
    "dcase": "dcase",
    "enabirds": "enabirds",
    "hiceas": "hiceas",
    "rfcx": "rfcx",
    "hainan": "hainan-gibbons",
    "gibbons": "gibbons",
}


# ============================================================================
# Dataset Classes
# ============================================================================

class BEANSDataset(Dataset):
    """Dataset wrapper for BEANS benchmark data."""

    def __init__(
        self,
        dataset_name: str,
        split: str = "test",
        max_length_seconds: float = 10.0,
        sample_rate: int = 16000,
    ):
        self.dataset_name = dataset_name
        self.max_length_seconds = max_length_seconds
        self.sample_rate = sample_rate
        self.max_samples = int(max_length_seconds * sample_rate)

        # Map to BEANS-Zero dataset name
        beans_name = DATASET_NAME_MAP.get(dataset_name, dataset_name)

        print(f"Loading BEANS dataset: {dataset_name} (mapped to: {beans_name})...")

        # Load from Hugging Face
        try:
            # Ensure cache directory exists
            HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

            # Verify cache configuration at runtime
            datasets_cache = os.environ.get("HF_DATASETS_CACHE", "")
            if not datasets_cache or str(Path.home()) in datasets_cache:
                raise RuntimeError(
                    f"HF_DATASETS_CACHE not properly configured: {datasets_cache}"
                )

            full_ds = load_dataset(
                "EarthSpeciesProject/BEANS-Zero",
                split=split,
                cache_dir=str(HF_CACHE_DIR),
                trust_remote_code=False,  # Security: don't execute remote code
            )

            # Filter to the specific dataset
            dataset_names = np.array(full_ds["dataset_name"])
            indices = np.where(dataset_names == beans_name)[0]

            if len(indices) == 0:
                # Try alternative names
                available = np.unique(dataset_names)
                print(f"  Dataset '{beans_name}' not found. Available: {list(available)}")
                # Try partial match
                for avail in available:
                    if dataset_name.lower() in avail.lower() or avail.lower() in dataset_name.lower():
                        print(f"  Using partial match: {avail}")
                        indices = np.where(dataset_names == avail)[0]
                        break

            if len(indices) > 0:
                self.data = full_ds.select(indices)
                print(f"  Loaded {len(self.data)} samples")
            else:
                print(f"  Warning: No samples found for {dataset_name}")
                self.data = None

        except Exception as e:
            print(f"  Error loading dataset: {e}")
            self.data = None

        # Build label mapping
        self.label_to_idx = {}
        self.idx_to_label = {}
        if self.data is not None:
            unique_labels = sorted(set(self.data["output"]))
            self.label_to_idx = {label: i for i, label in enumerate(unique_labels)}
            self.idx_to_label = {i: label for label, i in self.label_to_idx.items()}
            self.num_classes = len(unique_labels)
            print(f"  Found {self.num_classes} unique labels")

    def __len__(self):
        return len(self.data) if self.data is not None else 0

    def __getitem__(self, idx):
        if self.data is None:
            return None

        sample = self.data[idx]

        # Get audio
        audio = np.array(sample["audio"], dtype=np.float32)

        # Parse metadata for sample rate
        import json
        try:
            metadata = json.loads(sample["metadata"])
            orig_sr = metadata.get("sample_rate", 16000)
        except:
            orig_sr = 16000

        # Resample if needed
        if orig_sr != self.sample_rate:
            try:
                import resampy
                audio = resampy.resample(audio, orig_sr, self.sample_rate)
            except ImportError:
                # Simple decimation/interpolation as fallback
                ratio = self.sample_rate / orig_sr
                new_len = int(len(audio) * ratio)
                audio = np.interp(
                    np.linspace(0, len(audio) - 1, new_len),
                    np.arange(len(audio)),
                    audio
                )

        # Pad or truncate
        if len(audio) > self.max_samples:
            audio = audio[:self.max_samples]
        elif len(audio) < self.max_samples:
            audio = np.pad(audio, (0, self.max_samples - len(audio)))

        # Get label
        label_str = sample["output"]
        label = self.label_to_idx.get(label_str, 0)

        # Get task type
        task = sample.get("task", "classification")

        return {
            "audio": torch.from_numpy(audio),
            "label": label,
            "label_str": label_str,
            "task": task,
        }


def collate_beans(batch):
    """Collate function for BEANS dataset."""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    audio = torch.stack([b["audio"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch])
    label_strs = [b["label_str"] for b in batch]
    tasks = [b["task"] for b in batch]

    return {
        "audio": audio,
        "label": labels,
        "label_str": label_strs,
        "task": tasks,
    }


# ============================================================================
# Evaluation Functions
# ============================================================================

@torch.no_grad()
def extract_embeddings(encoder, dataloader, device):
    """Extract embeddings from audio encoder for all samples."""
    encoder.eval()

    all_embeddings = []
    all_labels = []
    all_label_strs = []

    for batch in tqdm(dataloader, desc="Extracting embeddings"):
        if batch is None:
            continue

        audio = batch["audio"].to(device)
        labels = batch["label"]
        label_strs = batch["label_str"]

        with autocast("cuda", dtype=torch.bfloat16):
            embeddings = encoder.encode(audio)

        all_embeddings.append(embeddings.float().cpu().numpy())
        all_labels.append(labels.numpy())
        all_label_strs.extend(label_strs)

    if len(all_embeddings) == 0:
        return None, None, None

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    return embeddings, labels, all_label_strs


def compute_linear_probe_accuracy(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
) -> float:
    """Train a linear probe and compute accuracy."""
    # Normalize embeddings
    train_norm = train_embeddings / (np.linalg.norm(train_embeddings, axis=1, keepdims=True) + 1e-8)
    test_norm = test_embeddings / (np.linalg.norm(test_embeddings, axis=1, keepdims=True) + 1e-8)

    # Train logistic regression
    clf = LogisticRegression(
        max_iter=1000,
        solver='lbfgs',
        multi_class='multinomial',
        n_jobs=-1,
        random_state=42
    )

    try:
        clf.fit(train_norm, train_labels)
        predictions = clf.predict(test_norm)
        accuracy = accuracy_score(test_labels, predictions)
    except Exception as e:
        print(f"  Warning: Linear probe failed: {e}")
        accuracy = 0.0

    return accuracy


def compute_map_score(
    embeddings: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
) -> float:
    """Compute mean Average Precision for multi-label detection."""
    # For detection tasks, we compute mAP using cosine similarity
    # between embeddings and class prototypes

    # Normalize embeddings
    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

    # Compute class prototypes (mean of embeddings per class)
    prototypes = np.zeros((num_classes, embeddings.shape[1]))
    for c in range(num_classes):
        class_mask = labels == c
        if class_mask.sum() > 0:
            prototypes[c] = embeddings_norm[class_mask].mean(axis=0)

    # Normalize prototypes
    prototypes = prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-8)

    # Compute similarity scores (embeddings x prototypes)
    scores = embeddings_norm @ prototypes.T  # (N, num_classes)

    # Convert labels to one-hot for AP calculation
    labels_onehot = np.zeros((len(labels), num_classes))
    labels_onehot[np.arange(len(labels)), labels] = 1

    # Compute AP for each class
    aps = []
    for c in range(num_classes):
        if labels_onehot[:, c].sum() > 0:
            ap = average_precision_score(labels_onehot[:, c], scores[:, c])
            aps.append(ap)

    if len(aps) == 0:
        return 0.0

    return np.mean(aps)


def evaluate_dataset(
    dataset_name: str,
    encoder: nn.Module,
    device: str,
    batch_size: int = 32,
    num_workers: int = 4,
    is_detection: bool = False,
) -> dict:
    """Evaluate encoder on a single BEANS dataset."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {dataset_name}")
    print(f"Task: {'Detection (mAP)' if is_detection else 'Classification (Accuracy)'}")
    print("="*60)

    # Load dataset
    dataset = BEANSDataset(dataset_name, split="test")

    if len(dataset) == 0:
        print(f"  Skipping {dataset_name}: no samples found")
        return {"dataset": dataset_name, "metric": 0.0, "num_samples": 0}

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_beans,
    )

    # Extract embeddings
    embeddings, labels, label_strs = extract_embeddings(encoder, dataloader, device)

    if embeddings is None:
        print(f"  Skipping {dataset_name}: embedding extraction failed")
        return {"dataset": dataset_name, "metric": 0.0, "num_samples": 0}

    print(f"  Extracted {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")

    num_classes = dataset.num_classes

    if is_detection:
        # Compute mAP for detection tasks
        metric = compute_map_score(embeddings, labels, num_classes)
        metric_name = "mAP"
    else:
        # For classification, split into train/test for linear probe
        # Use 80/20 split
        n = len(embeddings)
        train_size = int(0.8 * n)

        # Shuffle indices
        indices = np.random.permutation(n)
        train_idx = indices[:train_size]
        test_idx = indices[train_size:]

        train_embeddings = embeddings[train_idx]
        train_labels = labels[train_idx]
        test_embeddings = embeddings[test_idx]
        test_labels = labels[test_idx]

        metric = compute_linear_probe_accuracy(
            train_embeddings, train_labels,
            test_embeddings, test_labels
        )
        metric_name = "Accuracy"

    print(f"  {metric_name}: {metric*100:.2f}%")

    return {
        "dataset": dataset_name,
        "metric": metric,
        "metric_name": metric_name,
        "num_samples": len(embeddings),
        "num_classes": num_classes,
    }


# ============================================================================
# Results Formatting
# ============================================================================

def format_results_table(results: list, title: str, metric_name: str) -> str:
    """Format results as a table with Mean column."""
    # Header
    datasets = [r["dataset"] for r in results]
    header = "| Model | " + " | ".join(datasets) + " | Mean |"
    separator = "|" + "---|" * (len(datasets) + 2)

    # Values
    values = [r["metric"] * 100 for r in results]
    mean_value = np.mean(values) if values else 0.0

    row = "| NatureLM | " + " | ".join([f"{v:.2f}" for v in values]) + f" | {mean_value:.2f} |"

    table = f"\n## {title} ({metric_name})\n\n{header}\n{separator}\n{row}\n"
    return table


def print_summary_table(accuracy_results: list, map_results: list):
    """Print formatted summary tables."""
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)

    # Accuracy table
    print("\n## Classification Datasets (Accuracy %)")
    print("-" * 80)

    if accuracy_results:
        header = "Dataset".ljust(15) + "Accuracy".ljust(12) + "Samples".ljust(10) + "Classes"
        print(header)
        print("-" * 50)

        for r in accuracy_results:
            row = f"{r['dataset'].ljust(15)}{(r['metric']*100):.2f}%".ljust(12+15)
            row += f"{r['num_samples']}".ljust(10)
            row += f"{r.get('num_classes', 'N/A')}"
            print(row)

        mean_acc = np.mean([r["metric"] for r in accuracy_results]) * 100
        print("-" * 50)
        print(f"{'Mean'.ljust(15)}{mean_acc:.2f}%")

    # mAP table
    print("\n## Detection Datasets (mAP %)")
    print("-" * 80)

    if map_results:
        header = "Dataset".ljust(15) + "mAP".ljust(12) + "Samples".ljust(10) + "Classes"
        print(header)
        print("-" * 50)

        for r in map_results:
            row = f"{r['dataset'].ljust(15)}{(r['metric']*100):.2f}%".ljust(12+15)
            row += f"{r['num_samples']}".ljust(10)
            row += f"{r.get('num_classes', 'N/A')}"
            print(row)

        mean_map = np.mean([r["metric"] for r in map_results]) * 100
        print("-" * 50)
        print(f"{'Mean'.ljust(15)}{mean_map:.2f}%")

    print("\n" + "="*80)


def save_results_csv(accuracy_results: list, map_results: list, output_path: Path):
    """Save results to CSV file."""
    rows = []

    # Accuracy results
    for r in accuracy_results:
        rows.append({
            "dataset": r["dataset"],
            "task_type": "classification",
            "metric_name": "accuracy",
            "metric_value": r["metric"],
            "num_samples": r["num_samples"],
            "num_classes": r.get("num_classes", "N/A"),
        })

    # Add mean for accuracy
    if accuracy_results:
        mean_acc = np.mean([r["metric"] for r in accuracy_results])
        rows.append({
            "dataset": "Mean",
            "task_type": "classification",
            "metric_name": "accuracy",
            "metric_value": mean_acc,
            "num_samples": sum(r["num_samples"] for r in accuracy_results),
            "num_classes": "N/A",
        })

    # mAP results
    for r in map_results:
        rows.append({
            "dataset": r["dataset"],
            "task_type": "detection",
            "metric_name": "mAP",
            "metric_value": r["metric"],
            "num_samples": r["num_samples"],
            "num_classes": r.get("num_classes", "N/A"),
        })

    # Add mean for mAP
    if map_results:
        mean_map = np.mean([r["metric"] for r in map_results])
        rows.append({
            "dataset": "Mean",
            "task_type": "detection",
            "metric_name": "mAP",
            "metric_value": mean_map,
            "num_samples": sum(r["num_samples"] for r in map_results),
            "num_classes": "N/A",
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


# ============================================================================
# Main
# ============================================================================

def print_cache_configuration():
    """Print current HuggingFace cache configuration for verification."""
    print("\n" + "=" * 60)
    print("HuggingFace Cache Configuration")
    print("=" * 60)

    hf_vars = [
        "HF_HOME",
        "HF_DATASETS_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "HF_HUB_CACHE",
        "TRANSFORMERS_CACHE",
    ]

    home_dir = str(Path.home())
    all_safe = True

    for var in hf_vars:
        value = os.environ.get(var, "<not set>")
        is_safe = home_dir not in value if value != "<not set>" else False
        status = "[OK]" if is_safe else "[WARNING: contains home dir!]"
        if not is_safe and value != "<not set>":
            all_safe = False
        print(f"  {var}: {value} {status if value != '<not set>' else ''}")

    print(f"\nHome directory: {home_dir}")
    print(f"Project cache:  {HF_CACHE_DIR}")
    print(f"Cache exists:   {HF_CACHE_DIR.exists()}")

    if all_safe:
        print("\n[OK] All cache paths are outside home directory")
    else:
        print("\n[ERROR] Some cache paths point to home directory!")

    print("=" * 60 + "\n")
    return all_safe


def main():
    parser = argparse.ArgumentParser(description="Evaluate on BEANS benchmark datasets")

    # Model options
    parser.add_argument("--use_qformer", action="store_true", default=True,
                        help="Use Q-Former in NatureLM encoder")
    parser.add_argument("--pooling", type=str, default="mean",
                        choices=["mean", "max", "cls"])

    # Hardware
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)

    # Output
    parser.add_argument("--output_dir", type=str, default="./evaluation_results/beans_dataset_specific")

    # Dataset selection
    parser.add_argument("--accuracy_datasets", type=str, nargs="+",
                        default=ACCURACY_DATASETS,
                        help="Classification datasets to evaluate")
    parser.add_argument("--map_datasets", type=str, nargs="+",
                        default=MAP_DATASETS,
                        help="Detection datasets to evaluate")
    parser.add_argument("--skip_accuracy", action="store_true",
                        help="Skip classification datasets")
    parser.add_argument("--skip_map", action="store_true",
                        help="Skip detection datasets")

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Check dependencies
    if not HAS_HF_DATASETS:
        print("Error: huggingface datasets required. Install with: pip install datasets")
        return

    # Verify and display cache configuration
    cache_ok = print_cache_configuration()
    if not cache_ok:
        print("ERROR: Cache configuration unsafe. Aborting.")
        print("Please ensure HF cache is not in home directory.")
        return

    # Setup device
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load encoder
    print("\nLoading NatureLM audio encoder...")
    encoder = NatureLMAudioEncoder(use_qformer=args.use_qformer, pooling=args.pooling)
    encoder = encoder.to(device)
    encoder.eval()

    # Evaluate accuracy datasets
    accuracy_results = []
    if not args.skip_accuracy:
        print("\n" + "="*80)
        print("EVALUATING CLASSIFICATION DATASETS (Accuracy)")
        print("="*80)

        for dataset_name in args.accuracy_datasets:
            result = evaluate_dataset(
                dataset_name=dataset_name,
                encoder=encoder,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                is_detection=False,
            )
            accuracy_results.append(result)

    # Evaluate mAP datasets
    map_results = []
    if not args.skip_map:
        print("\n" + "="*80)
        print("EVALUATING DETECTION DATASETS (mAP)")
        print("="*80)

        for dataset_name in args.map_datasets:
            result = evaluate_dataset(
                dataset_name=dataset_name,
                encoder=encoder,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                is_detection=True,
            )
            map_results.append(result)

    # Print summary
    print_summary_table(accuracy_results, map_results)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"beans_results_{timestamp}.csv"
    save_results_csv(accuracy_results, map_results, csv_path)

    # Create markdown summary
    md_content = "# BEANS Benchmark Evaluation Results\n\n"
    md_content += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    md_content += f"Model: NatureLM-audio\n"
    md_content += f"Pooling: {args.pooling}\n"
    md_content += f"Q-Former: {args.use_qformer}\n\n"

    if accuracy_results:
        md_content += format_results_table(accuracy_results, "Classification Datasets", "Accuracy %")

    if map_results:
        md_content += format_results_table(map_results, "Detection Datasets", "mAP %")

    md_path = output_dir / f"beans_results_{timestamp}.md"
    with open(md_path, "w") as f:
        f.write(md_content)

    print(f"\nMarkdown summary saved to: {md_path}")
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
