"""
Dataset module for bio-acoustic classification training.

Provides dataloaders for BEANS and Birdset dataset families.
"""

from .base import BaseAudioDataset, collate_fn
from .cbi import CBIDataset
from .birdset import BirdsetDataset, BIRDSET_DATASETS

# BEANS dataset identifiers
BEANS_DATASETS = ["cbi"]

__all__ = [
    "BaseAudioDataset",
    "collate_fn",
    "CBIDataset",
    "BirdsetDataset",
    "BIRDSET_DATASETS",
    "BEANS_DATASETS",
]
