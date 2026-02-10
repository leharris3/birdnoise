"""
Birdset dataset loaders.

Supports: PER, NES, UHH, HSN, SSW, SNE
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .base import (
    BaseAudioDataset,
    parse_time_to_hour,
    safe_float,
)

# Available Birdset datasets
BIRDSET_DATASETS = ["PER", "NES", "UHH", "HSN", "SSW", "SNE"]


def parse_birdset_filename(filename: str) -> Dict[str, Any]:
    """
    Extract date and time from Birdset filename.

    Format: {DATASET}_{ID}_{YYYYMMDD}_{HHMMSS}_{START}_{END}.ogg
    Example: HSN_001_20150708_061805_000_005.ogg

    Returns:
        Dict with day_of_year and hour, or defaults if parsing fails
    """
    # Pattern: DATASET_ID_DATE_TIME_START_END.ogg
    # DATE is YYYYMMDD, TIME is HHMMSS
    match = re.match(r'[A-Z]+_\d+(?:_S\d+)?_(\d{8})_(\d{6})Z?_\d+_\d+\.ogg', filename)
    if match:
        date_str, time_str = match.groups()
        try:
            dt = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
            return {
                "day_of_year": dt.timetuple().tm_yday,
                "hour": dt.hour,
            }
        except ValueError:
            pass

    return {"day_of_year": 182, "hour": 12}


class BirdsetDataset(BaseAudioDataset):
    """
    Birdset dataset loader.

    Directory structure:
        Data/birdset/{DATASET}/
        ├── audio/                          # Audio files (.ogg)
        ├── {DATASET}/                      # Metadata directory
        │   ├── {DATASET}_metadata_train.parquet
        │   ├── {DATASET}_metadata_test.parquet
        │   └── {DATASET}_metadata_test_5s.parquet
        └── metadata/                       # Additional metadata

    The audio directory contains 5-second test segments with filenames like:
        {DATASET}_{ID}_{YYYYMMDD}_{HHMMSS}_{START}_{END}.ogg

    Note: Training audio (full XC recordings) is not distributed in the
    standard download. We use the test_5s split for evaluation.
    """

    dataset_type = "birdset"

    # Default max samples for SSW dataset (20k random subset)
    SSW_DEFAULT_MAX_SAMPLES = 20000

    def __init__(
        self,
        data_dir: str,
        dataset_name: str,
        split: str = "test_5s",
        sample_rate: int = 16000,
        max_length_seconds: float = 5.0,
        label_to_idx: Optional[Dict[str, int]] = None,
        multilabel: bool = False,
        max_samples: Optional[int] = None,
        subset_seed: int = 42,
    ):
        """
        Args:
            data_dir: Base path to birdset data (e.g., Data/birdset)
            dataset_name: Name of dataset (e.g., "HSN", "PER")
            split: Dataset split ("train", "test", "test_5s")
            sample_rate: Target sample rate
            max_length_seconds: Maximum audio length in seconds
            label_to_idx: Mapping from ebird codes to indices
            multilabel: If True, use multilabel targets
            max_samples: Maximum number of samples to use (random subset). If None,
                         uses SSW_DEFAULT_MAX_SAMPLES (20k) for SSW dataset, all samples otherwise.
            subset_seed: Random seed for reproducible subset selection (default: 42)
        """
        super().__init__(sample_rate, max_length_seconds, label_to_idx)

        self.data_dir = Path(data_dir)
        self.dataset_name = dataset_name.upper()  # Instance attribute for per-sample tracking
        self.split = split
        self.multilabel = multilabel
        self.subset_seed = subset_seed

        if self.dataset_name not in BIRDSET_DATASETS:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available: {BIRDSET_DATASETS}"
            )

        # Set paths
        self.dataset_dir = self.data_dir / self.dataset_name
        self.audio_dir = self.dataset_dir / "audio"
        self.metadata_dir = self.dataset_dir / self.dataset_name

        # Load metadata
        parquet_path = self.metadata_dir / f"{self.dataset_name}_metadata_{split}.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"Metadata not found: {parquet_path}")

        self.metadata_df = pd.read_parquet(parquet_path)

        # Apply random subset for SSW dataset (default 20k) or custom max_samples
        # max_samples: None = use default (20k for SSW), -1 = use all, positive = use that value
        if max_samples is None and self.dataset_name == "SSW":
            max_samples = self.SSW_DEFAULT_MAX_SAMPLES
        elif max_samples == -1:
            max_samples = None  # -1 means use all samples

        if max_samples is not None and max_samples > 0 and len(self.metadata_df) > max_samples:
            print(f"Subsampling {self.dataset_name} from {len(self.metadata_df)} to {max_samples} samples (seed={subset_seed})")
            rng = np.random.RandomState(subset_seed)
            subset_indices = rng.choice(len(self.metadata_df), size=max_samples, replace=False)
            subset_indices = np.sort(subset_indices)  # Keep sorted for consistent ordering
            self.metadata_df = self.metadata_df.iloc[subset_indices].reset_index(drop=True)
            self.original_indices = subset_indices
        else:
            self.original_indices = np.arange(len(self.metadata_df))

        # For test/test_5s, audio files follow a different naming scheme
        # Build mapping from index to audio file
        self._build_audio_mapping()

        print(f"BirdsetDataset ({self.dataset_name}/{split}): {len(self)} samples")

    def _build_audio_mapping(self):
        """Build mapping from dataset index to audio file.

        Handles both full datasets and subsampled datasets. For subsampled
        datasets, uses self.original_indices to select the correct audio files.
        """
        self.audio_files = []

        if "filepath" in self.metadata_df.columns:
            # Train split has filepath column
            for fp in self.metadata_df["filepath"]:
                audio_path = self.audio_dir / fp
                if audio_path.exists():
                    self.audio_files.append(audio_path)
                else:
                    self.audio_files.append(None)
        else:
            # Test splits: audio files are named sequentially
            # Get all audio files and sort them
            if self.audio_dir.exists():
                all_audio = sorted(self.audio_dir.glob("*.ogg"))

                if len(all_audio) > 0:
                    # Select audio files based on original_indices (handles subsampling)
                    for orig_idx in self.original_indices:
                        if orig_idx < len(all_audio):
                            self.audio_files.append(all_audio[orig_idx])
                        else:
                            self.audio_files.append(None)
                else:
                    self.audio_files = [None] * len(self.metadata_df)
            else:
                self.audio_files = [None] * len(self.metadata_df)

        # Count valid files
        valid_count = sum(1 for f in self.audio_files if f is not None)
        if valid_count < len(self.metadata_df):
            print(f"Warning: Only {valid_count}/{len(self.metadata_df)} audio files found")

    def __len__(self) -> int:
        return len(self.metadata_df)

    def _get_audio_path(self, idx: int) -> Path:
        audio_file = self.audio_files[idx]
        if audio_file is None:
            # Return a dummy path that will fail to load
            return Path("/nonexistent/audio.ogg")
        return audio_file

    def _get_label(self, idx: int) -> str:
        """Get primary label for a sample.

        For test_5s splits, ebird_code may be None - use first label from multilabel.
        """
        row = self.metadata_df.iloc[idx]

        # Try ebird_code first
        label = row.get("ebird_code")
        if label is not None and label == label:  # Check for NaN
            return str(label)

        # Fall back to first label from multilabel column
        multilabel = self._get_multilabel_raw(idx)
        if multilabel and len(multilabel) > 0:
            return str(multilabel[0])

        return "unknown"

    def _get_multilabel_raw(self, idx: int) -> List[str]:
        """Get raw multilabel list for a sample."""
        row = self.metadata_df.iloc[idx]

        # Try ebird_code_multilabel first
        if "ebird_code_multilabel" in self.metadata_df.columns:
            labels = row["ebird_code_multilabel"]
        elif "ebird_code_multiclass" in self.metadata_df.columns:
            labels = row["ebird_code_multiclass"]
        else:
            label = row.get("ebird_code")
            return [str(label)] if label is not None else []

        # Handle numpy arrays
        if isinstance(labels, np.ndarray):
            labels = labels.tolist()

        # Handle None or NaN
        if labels is None:
            return []

        if isinstance(labels, str):
            return [labels]

        if isinstance(labels, list):
            return [str(l) for l in labels if l is not None]

        return []

    def _get_multilabel(self, idx: int) -> List[str]:
        """Get multilabel targets for a sample."""
        return self._get_multilabel_raw(idx)

    def _get_metadata(self, idx: int) -> Dict[str, Any]:
        row = self.metadata_df.iloc[idx]

        # Get lat/long from metadata
        lat = safe_float(row.get("lat"), 0.0)
        lon = safe_float(row.get("long"), 0.0)

        # Try to get time from local_time column
        local_time = row.get("local_time", None)
        hour = parse_time_to_hour(local_time)

        # Get day of year from filename if available
        audio_file = self.audio_files[idx]
        if audio_file is not None:
            parsed = parse_birdset_filename(audio_file.name)
            day_of_year = parsed["day_of_year"]
            # Override hour from filename if local_time not available
            if local_time is None:
                hour = parsed["hour"]
        else:
            day_of_year = 182

        return {
            "latitude": lat,
            "longitude": lon,
            "day_of_year": day_of_year,
            "hour": hour,
            "ebird_code": row.get("ebird_code", ""),
        }

    def _get_original_index(self, idx: int) -> int:
        return int(self.original_indices[idx])

    def get_ebird_codes(self) -> List[str]:
        """Return sorted list of eBird codes in dataset."""
        codes = self.metadata_df["ebird_code"].dropna().unique()
        return sorted([str(c) for c in codes if c is not None])

    def get_all_labels(self) -> List[str]:
        """Return all unique labels (eBird codes).

        Handles both single-label (ebird_code) and multilabel columns.
        For test_5s splits where ebird_code may be None, uses multilabel column.
        """
        all_labels = set()

        # Collect from ebird_code column
        for code in self.metadata_df["ebird_code"]:
            if code is not None and code == code:  # Skip NaN
                all_labels.add(str(code))

        # Also collect from multilabel columns
        for col in ["ebird_code_multilabel", "ebird_code_multiclass"]:
            if col in self.metadata_df.columns:
                for labels in self.metadata_df[col]:
                    if labels is None:
                        continue
                    if isinstance(labels, np.ndarray):
                        labels = labels.tolist()
                    if isinstance(labels, list):
                        for l in labels:
                            if l is not None:
                                all_labels.add(str(l))
                    elif labels is not None:
                        all_labels.add(str(labels))

        # Remove 'unknown' if it was added
        all_labels.discard("unknown")

        return sorted(all_labels) if all_labels else ["unknown"]

    def get_num_classes(self) -> int:
        """Return number of classes in dataset."""
        return len(self.get_all_labels())


def load_birdset_dataset(
    base_dir: str,
    dataset_name: str,
    split: str = "test_5s",
    **kwargs,
) -> BirdsetDataset:
    """
    Convenience function to load a Birdset dataset.

    Args:
        base_dir: Base path to birdset data directory
        dataset_name: Name of dataset (e.g., "HSN", "PER")
        split: Dataset split
        **kwargs: Additional arguments passed to BirdsetDataset

    Returns:
        BirdsetDataset instance
    """
    return BirdsetDataset(
        data_dir=base_dir,
        dataset_name=dataset_name,
        split=split,
        **kwargs,
    )


def load_multiple_birdset_datasets(
    base_dir: str,
    dataset_names: List[str],
    split: str = "test_5s",
    **kwargs,
) -> Dict[str, BirdsetDataset]:
    """
    Load multiple Birdset datasets.

    Args:
        base_dir: Base path to birdset data directory
        dataset_names: List of dataset names
        split: Dataset split
        **kwargs: Additional arguments passed to BirdsetDataset

    Returns:
        Dictionary mapping dataset names to BirdsetDataset instances
    """
    datasets = {}
    for name in dataset_names:
        try:
            datasets[name] = load_birdset_dataset(
                base_dir, name, split, **kwargs
            )
        except FileNotFoundError as e:
            print(f"Warning: Could not load {name}: {e}")
    return datasets
