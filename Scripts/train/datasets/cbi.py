"""
CBI (Cornell Bird Identification) dataset loader for BEANS benchmark.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .base import (
    BaseAudioDataset,
    parse_date_to_day_of_year,
    parse_time_to_hour,
    safe_float,
)


class CBIDataset(BaseAudioDataset):
    """
    CBI dataset from BEANS benchmark.

    Directory structure:
        Data/beans/cbi/
        ├── train_audio/          # Audio files organized by ebird_code
        │   └── {ebird_code}/
        │       └── {filename}.ogg
        ├── train.csv             # Training metadata
        ├── test.csv              # Test metadata
        └── priors/               # Prior probability data

    CSV columns:
        - filename: Audio file name (e.g., XC134874.mp3)
        - ebird_code: Species code directory (e.g., aldfly)
        - species: Full species name for labeling
        - latitude, longitude: Recording location
        - date: Recording date (YYYY-MM-DD)
        - time: Recording time (HH:MM)
    """

    dataset_type = "beans"
    dataset_name = "cbi"

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        sample_rate: int = 16000,
        max_length_seconds: float = 10.0,
        label_to_idx: Optional[Dict[str, int]] = None,
        filter_missing: bool = True,
    ):
        """
        Args:
            data_dir: Path to CBI data directory (e.g., Data/beans/cbi)
            split: Dataset split ("train" or "test")
            sample_rate: Target sample rate
            max_length_seconds: Maximum audio length in seconds
            label_to_idx: Mapping from species names to indices
            filter_missing: If True, filter out samples with missing audio files
        """
        super().__init__(sample_rate, max_length_seconds, label_to_idx)

        self.data_dir = Path(data_dir)
        self.split = split
        self.audio_dir = self.data_dir / "train_audio"

        # Load metadata
        csv_path = self.data_dir / f"{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {csv_path}")

        self.metadata_df = pd.read_csv(csv_path)
        self.original_indices = self.metadata_df.index.values.copy()

        # Filter to existing audio files if requested
        if filter_missing:
            valid_mask = self._check_audio_files()
            self.metadata_df = self.metadata_df[valid_mask].reset_index(drop=True)
            self.original_indices = self.original_indices[valid_mask]

        print(f"CBIDataset ({split}): {len(self.metadata_df)} samples")

    def _check_audio_files(self) -> List[bool]:
        """Check which audio files exist."""
        valid = []
        for _, row in self.metadata_df.iterrows():
            audio_path = self._build_audio_path(row)
            valid.append(audio_path.exists())
        return valid

    def _build_audio_path(self, row: pd.Series) -> Path:
        """Build audio file path from metadata row."""
        # Files may be .ogg despite .mp3 extension in CSV
        filename = row["filename"]
        ebird_code = row["ebird_code"]
        audio_path = self.audio_dir / ebird_code / filename

        # Try .ogg if .mp3 doesn't exist
        if not audio_path.exists():
            audio_path = Path(str(audio_path).replace(".mp3", ".ogg"))

        return audio_path

    def __len__(self) -> int:
        return len(self.metadata_df)

    def _get_audio_path(self, idx: int) -> Path:
        row = self.metadata_df.iloc[idx]
        return self._build_audio_path(row)

    def _get_label(self, idx: int) -> str:
        return self.metadata_df.iloc[idx]["species"]

    def _get_metadata(self, idx: int) -> Dict[str, Any]:
        row = self.metadata_df.iloc[idx]

        # Parse date
        date_str = row.get("date", "")
        day_of_year = parse_date_to_day_of_year(date_str)

        # Parse time
        time_val = row.get("time", "12:00")
        hour = parse_time_to_hour(time_val)

        return {
            "latitude": safe_float(row.get("latitude"), 0.0),
            "longitude": safe_float(row.get("longitude"), 0.0),
            "day_of_year": day_of_year,
            "hour": hour,
            "date": str(date_str),
            "ebird_code": row.get("ebird_code", ""),
        }

    def _get_original_index(self, idx: int) -> int:
        return int(self.original_indices[idx])

    def get_species_list(self) -> List[str]:
        """Return sorted list of species in dataset."""
        return sorted(self.metadata_df["species"].unique())

    def get_ebird_codes(self) -> List[str]:
        """Return sorted list of eBird codes in dataset."""
        return sorted(self.metadata_df["ebird_code"].unique())

    def get_all_labels(self) -> List[str]:
        """Return sorted list of all unique labels (species names)."""
        return self.get_species_list()

    @staticmethod
    def get_priors_cache_path(data_dir: str) -> Optional[Path]:
        """Find priors cache file if it exists."""
        data_dir = Path(data_dir)

        # Look for H5 cache files
        h5_files = list(data_dir.glob("priors_cache*.h5"))
        if h5_files:
            return h5_files[0]

        # Look for NPZ files
        npz_files = list(data_dir.glob("priors_cache*.npz"))
        if npz_files:
            return npz_files[0]

        return None
