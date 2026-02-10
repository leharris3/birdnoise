"""
Base classes and utilities for audio datasets.
"""

import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

# Suppress audio loading warnings
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    import resampy
    HAS_RESAMPY = True
except ImportError:
    HAS_RESAMPY = False


def load_audio(
    path: str,
    target_sr: int = 16000,
    max_samples: Optional[int] = None,
) -> Optional[np.ndarray]:
    """
    Load and preprocess audio file.

    Args:
        path: Path to audio file
        target_sr: Target sample rate
        max_samples: Maximum number of samples (truncate if longer, pad if shorter)

    Returns:
        Audio as numpy array, or None if loading fails
    """
    if not HAS_SOUNDFILE:
        raise ImportError("soundfile is required for audio loading")

    try:
        audio, sr = sf.read(path)
    except Exception as e:
        warnings.warn(f"Failed to load audio: {path} - {e}")
        return None

    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Resample if needed
    if sr != target_sr:
        if not HAS_RESAMPY:
            raise ImportError("resampy is required for audio resampling")
        audio = resampy.resample(audio, sr, target_sr)

    # Pad or truncate if max_samples specified
    if max_samples is not None:
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        elif len(audio) < max_samples:
            audio = np.pad(audio, (0, max_samples - len(audio)))

    return audio.astype(np.float32)


def parse_date_to_day_of_year(date_str: str, default: int = 182) -> int:
    """Parse date string to day of year."""
    if not date_str or date_str in ("", "nan", "None", None):
        return default

    try:
        import pandas as pd
        date = pd.to_datetime(date_str)
        return date.timetuple().tm_yday
    except Exception:
        return default


def parse_time_to_hour(time_str: Any, default: int = 12) -> int:
    """Parse time string to hour."""
    if time_str is None or (isinstance(time_str, float) and np.isnan(time_str)):
        return default

    try:
        if isinstance(time_str, str) and ":" in time_str:
            return int(time_str.split(":")[0])
        return int(time_str)
    except Exception:
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float."""
    if value is None:
        return default
    try:
        f = float(value)
        if np.isnan(f):
            return default
        return f
    except (ValueError, TypeError):
        return default


class BaseAudioDataset(Dataset, ABC):
    """
    Abstract base class for audio datasets.

    All datasets return batches with structure:
    {
        "audio": torch.Tensor,           # Shape: (batch_size, num_samples)
        "label": torch.Tensor,           # Shape: (batch_size,) - class indices
        "multilabel": torch.Tensor,      # Shape: (batch_size, num_classes) - multi-hot targets
        "metadata": List[dict],          # Each dict has: latitude, longitude, day_of_year, hour
        "sample_indices": List[int],     # Original indices for prior cache lookup
        "dataset_name": str,             # Dataset identifier for per-dataset metrics
    }
    """

    # Dataset type identifier: "beans" or "birdset"
    dataset_type: str = "unknown"
    # Dataset name (e.g., "cbi", "HSN", "PER")
    dataset_name: str = "unknown"

    def __init__(
        self,
        sample_rate: int = 16000,
        max_length_seconds: float = 5.0,
        label_to_idx: Optional[Dict[str, int]] = None,
    ):
        """
        Args:
            sample_rate: Target sample rate for audio
            max_length_seconds: Maximum audio length in seconds
            label_to_idx: Mapping from label strings to indices
        """
        self.sample_rate = sample_rate
        self.max_samples = int(max_length_seconds * sample_rate)
        self.label_to_idx = label_to_idx or {}
        self._cache: Dict[int, Dict] = {}
        self._num_classes: Optional[int] = None

    @abstractmethod
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        pass

    @abstractmethod
    def _get_audio_path(self, idx: int) -> Path:
        """Return path to audio file for given index."""
        pass

    @abstractmethod
    def _get_label(self, idx: int) -> str:
        """Return label string for given index."""
        pass

    @abstractmethod
    def _get_metadata(self, idx: int) -> Dict[str, Any]:
        """Return metadata dict for given index."""
        pass

    @abstractmethod
    def _get_original_index(self, idx: int) -> int:
        """Return original dataset index for prior cache lookup."""
        pass

    def _get_multilabel(self, idx: int) -> List[str]:
        """
        Return list of all labels for given index (for multi-label classification).

        Default implementation returns single label. Override in subclasses
        for datasets with multi-label annotations.
        """
        return [self._get_label(idx)]

    def get_num_classes(self) -> int:
        """Return number of classes in the dataset."""
        if self._num_classes is not None:
            return self._num_classes
        if self.label_to_idx:
            return len(self.label_to_idx)
        return len(self.get_all_labels())

    def set_num_classes(self, num_classes: int):
        """Set the number of classes (for unified label space)."""
        self._num_classes = num_classes

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample."""
        if idx in self._cache:
            return self._cache[idx]

        audio_path = self._get_audio_path(idx)
        audio = load_audio(str(audio_path), self.sample_rate, self.max_samples)

        if audio is None:
            # Return zeros for failed loads
            audio = np.zeros(self.max_samples, dtype=np.float32)
            warnings.warn(f"Failed to load audio at index {idx}, using zeros")

        label_str = self._get_label(idx)
        label_idx = self.label_to_idx.get(label_str, 0)

        metadata = self._get_metadata(idx)

        # Build multi-hot label vector
        num_classes = self.get_num_classes()
        multilabel_vec = np.zeros(num_classes, dtype=np.float32)
        for lbl in self._get_multilabel(idx):
            if lbl in self.label_to_idx:
                multilabel_vec[self.label_to_idx[lbl]] = 1.0

        sample = {
            "audio": torch.from_numpy(audio),
            "label": label_idx,
            "multilabel": torch.from_numpy(multilabel_vec),
            "metadata": {
                "latitude": safe_float(metadata.get("latitude"), 0.0),
                "longitude": safe_float(metadata.get("longitude"), 0.0),
                "day_of_year": metadata.get("day_of_year", 182),
                "hour": metadata.get("hour", 12),
            },
            "sample_idx": self._get_original_index(idx),
            "dataset_name": self.dataset_name,
        }

        self._cache[idx] = sample
        return sample

    def get_all_labels(self) -> List[str]:
        """Return list of all unique labels in dataset."""
        labels = set()
        for i in range(len(self)):
            labels.add(self._get_label(i))
        return sorted(labels)

    def build_label_mapping(self) -> Dict[str, int]:
        """Build and return label to index mapping."""
        labels = self.get_all_labels()
        self.label_to_idx = {label: i for i, label in enumerate(labels)}
        return self.label_to_idx

    def clear_cache(self):
        """Clear the sample cache."""
        self._cache.clear()


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    Collate function for DataLoader.

    Args:
        batch: List of sample dictionaries

    Returns:
        Collated batch dictionary
    """
    return {
        "audio": torch.stack([b["audio"] for b in batch]),
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "multilabel": torch.stack([b["multilabel"] for b in batch]),
        "metadata": [b["metadata"] for b in batch],
        "sample_indices": [b["sample_idx"] for b in batch],
        "dataset_names": [b["dataset_name"] for b in batch],
    }
