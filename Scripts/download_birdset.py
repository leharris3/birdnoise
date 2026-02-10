#!/usr/bin/env python3
"""
Download the BirdSet dataset from HuggingFace.

Downloads all BirdSet subsets to the Data/birdset directory with:
- Resume capability for interrupted downloads
- Verification of downloaded files
- Progress logging to file and console

BirdSet subsets:
- HSN: High Sierra Nevada
- NBP: NorthEast Birds Prediction
- POW: Powdermill Nature Reserve
- PER: Peru
- NES: Northeastern USA/Southeast Canada
- UHH: University of Hawaii at Hilo
- SSW: South Sweden
- SNE: Sierra Nevada East
- XCM: Xeno-Canto Medium (for pretraining)
- XCL: Xeno-Canto Large (for pretraining)

Reference: https://huggingface.co/datasets/DBD-research-group/BirdSet

Note: Uses direct download from the 'data' branch since datasets>=3.0.0
no longer supports loading scripts with trust_remote_code.
"""

import os
import sys
import json
import logging
import argparse
import tarfile
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

# ============================================================================
# HuggingFace Cache Configuration - MUST be set BEFORE importing HF libraries
# ============================================================================

_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
_HF_CACHE_DIR = _PROJECT_ROOT / ".cache" / "huggingface"


def _setup_hf_cache_environment():
    """Configure HuggingFace environment variables to use project-local cache."""
    cache_dir = str(_HF_CACHE_DIR)
    _HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    hf_env_vars = {
        "HF_HOME": cache_dir,
        "HF_DATASETS_CACHE": str(_HF_CACHE_DIR / "datasets"),
        "HUGGINGFACE_HUB_CACHE": str(_HF_CACHE_DIR / "hub"),
        "HF_HUB_CACHE": str(_HF_CACHE_DIR / "hub"),
        "TRANSFORMERS_CACHE": str(_HF_CACHE_DIR / "transformers"),
    }

    for var, value in hf_env_vars.items():
        os.environ[var] = value


# Setup cache BEFORE importing HF libraries
_setup_hf_cache_environment()

from huggingface_hub import HfApi, hf_hub_download
import pandas as pd
from tqdm import tqdm

# ============================================================================
# Constants
# ============================================================================

BIRDSET_REPO = "DBD-research-group/BirdSet"
BIRDSET_DATA_BRANCH = "data"

# All available BirdSet subsets
BIRDSET_SUBSETS = [
    "HSN",  # High Sierra Nevada
    "NBP",  # NorthEast Birds Prediction
    "POW",  # Powdermill Nature Reserve
    "PER",  # Peru
    "NES",  # Northeastern USA/Southeast Canada
    "UHH",  # University of Hawaii at Hilo
    "SSW",  # South Sweden
    "SNE",  # Sierra Nevada East
    "XCM",  # Xeno-Canto Medium (pretraining)
    "XCL",  # Xeno-Canto Large (pretraining)
]

# Subsets that have train/test splits vs single split
SUBSETS_WITH_SPLITS = ["HSN", "NBP", "POW", "PER", "NES", "UHH", "SSW", "SNE"]
SUBSETS_SINGLE_SPLIT = ["XCM", "XCL"]

# Default output directory
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "Data" / "birdset"

# Progress tracking file
PROGRESS_FILE = "download_progress.json"

# ============================================================================
# Logging Setup
# ============================================================================


def setup_logging(output_dir: Path, verbose: bool = False) -> logging.Logger:
    """Configure logging to both file and console."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"download_{timestamp}.log"

    # Create logger
    logger = logging.getLogger("birdset_downloader")
    logger.setLevel(logging.DEBUG)  # Set to DEBUG; handlers control actual output level

    # Clear existing handlers to prevent duplicate messages on repeated calls
    if logger.handlers:
        logger.handlers.clear()

    # File handler (always verbose)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Log file: {log_file}")

    return logger


# ============================================================================
# Progress Tracking
# ============================================================================


class ProgressTracker:
    """Track download progress for resume capability."""

    def __init__(self, output_dir: Path):
        self.progress_file = output_dir / PROGRESS_FILE
        self.progress = self._load_progress()

    def _load_progress(self) -> dict:
        """Load existing progress from file."""
        default = {"completed_subsets": [], "failed_subsets": {}, "verified": []}
        if self.progress_file.exists():
            try:
                with open(self.progress_file, "r") as f:
                    data = json.load(f)
                # Validate expected keys exist
                for key in default:
                    if key not in data:
                        data[key] = default[key]
                return data
            except (json.JSONDecodeError, OSError, ValueError):
                return default
        return default

    def save(self):
        """Save progress to file."""
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.progress_file, "w") as f:
            json.dump(self.progress, f, indent=2)

    def mark_completed(self, subset: str):
        """Mark a subset as successfully downloaded."""
        if subset not in self.progress["completed_subsets"]:
            self.progress["completed_subsets"].append(subset)
        if subset in self.progress["failed_subsets"]:
            del self.progress["failed_subsets"][subset]
        self.save()

    def mark_failed(self, subset: str, error: str):
        """Mark a subset as failed with error message."""
        self.progress["failed_subsets"][subset] = {
            "error": error,
            "timestamp": datetime.now().isoformat(),
        }
        self.save()

    def mark_verified(self, subset: str):
        """Mark a subset as verified."""
        if subset not in self.progress["verified"]:
            self.progress["verified"].append(subset)
        self.save()

    def is_completed(self, subset: str) -> bool:
        """Check if a subset has been downloaded."""
        return subset in self.progress["completed_subsets"]

    def is_verified(self, subset: str) -> bool:
        """Check if a subset has been verified."""
        return subset in self.progress["verified"]

    def get_pending_subsets(self, subsets: list) -> list:
        """Get list of subsets that haven't been downloaded yet."""
        return [s for s in subsets if not self.is_completed(s)]

    def get_summary(self) -> dict:
        """Get summary of progress."""
        return {
            "completed": len(self.progress["completed_subsets"]),
            "failed": len(self.progress["failed_subsets"]),
            "verified": len(self.progress["verified"]),
        }


# ============================================================================
# Dataset Download Functions
# ============================================================================


def get_available_subsets(logger: logging.Logger) -> list:
    """Fetch available subset configurations from HuggingFace data branch."""
    logger.info(f"Fetching available configurations from {BIRDSET_REPO} (branch: {BIRDSET_DATA_BRANCH})...")
    try:
        api = HfApi()
        files = api.list_repo_files(
            BIRDSET_REPO,
            repo_type="dataset",
            revision=BIRDSET_DATA_BRANCH
        )
        # Extract subset names from directory structure
        subsets = set()
        for f in files:
            if "/" in f:
                subset = f.split("/")[0]
                if subset in BIRDSET_SUBSETS:
                    subsets.add(subset)
        configs = sorted(list(subsets))
        logger.info(f"Found {len(configs)} configurations: {configs}")
        return configs
    except Exception as e:
        logger.warning(f"Could not fetch configurations: {e}")
        logger.info(f"Using default subset list: {BIRDSET_SUBSETS}")
        return BIRDSET_SUBSETS


def _get_subset_files(subset: str, logger: logging.Logger) -> Dict[str, List[str]]:
    """Get list of files for a subset from the data branch."""
    api = HfApi()
    all_files = api.list_repo_files(
        BIRDSET_REPO,
        repo_type="dataset",
        revision=BIRDSET_DATA_BRANCH
    )

    subset_files = [f for f in all_files if f.startswith(f"{subset}/")]

    result = {
        "metadata": [],
        "audio_archives": [],
    }

    for f in subset_files:
        if f.endswith(".parquet"):
            result["metadata"].append(f)
        elif f.endswith(".tar.gz"):
            result["audio_archives"].append(f)

    logger.debug(f"Found {len(result['metadata'])} metadata files and {len(result['audio_archives'])} audio archives for {subset}")
    return result


def _extract_tar_gz(tar_path: Path, output_dir: Path, logger: logging.Logger) -> Path:
    """Extract tar.gz file to output directory, flattening the structure."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already extracted
    if output_dir.exists() and any(output_dir.iterdir()):
        logger.debug(f"Already extracted: {output_dir}")
        return output_dir

    logger.debug(f"Extracting {tar_path.name} to {output_dir}")
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.isfile():
                # Flatten: extract just the filename without directory structure
                # Use extractfile + write to avoid mutating member and for Python <3.12 compat
                filename = os.path.basename(member.name)
                # Security check: skip files with suspicious names
                if not filename or filename.startswith('.') or '/' in filename or '\\' in filename:
                    logger.debug(f"Skipping suspicious filename: {member.name}")
                    continue
                file_obj = tar.extractfile(member)
                if file_obj is not None:
                    dest_path = output_dir / filename
                    with open(dest_path, 'wb') as out_file:
                        out_file.write(file_obj.read())

    return output_dir


def download_subset(
    subset: str,
    output_dir: Path,
    logger: logging.Logger,
    num_proc: int = 4,
    keep_archives: bool = False,
) -> bool:
    """
    Download a single BirdSet subset directly from the data branch.

    Args:
        subset: Name of the subset to download
        output_dir: Directory to save the dataset
        logger: Logger instance
        num_proc: Number of processes for parallel download (unused, kept for API compat)
        keep_archives: If True, keep the downloaded tar.gz files after extraction

    Returns:
        True if successful, False otherwise
    """
    subset_dir = output_dir / subset
    subset_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading subset '{subset}' to {subset_dir}...")

    try:
        # Get list of files for this subset
        files = _get_subset_files(subset, logger)

        if not files["metadata"]:
            logger.error(f"No metadata files found for '{subset}'")
            return False

        # Download metadata files
        logger.info(f"Downloading {len(files['metadata'])} metadata file(s)...")
        metadata_dir = subset_dir / "metadata"
        metadata_dir.mkdir(exist_ok=True)

        for meta_file in files["metadata"]:
            logger.debug(f"Downloading {meta_file}...")
            local_path = hf_hub_download(
                BIRDSET_REPO,
                meta_file,
                repo_type="dataset",
                revision=BIRDSET_DATA_BRANCH,
                local_dir=subset_dir,
            )
            logger.debug(f"Downloaded to {local_path}")

        # Download and extract audio archives
        if files["audio_archives"]:
            logger.info(f"Downloading and extracting {len(files['audio_archives'])} audio archive(s)...")
            audio_dir = subset_dir / "audio"
            audio_dir.mkdir(exist_ok=True)
            archives_dir = subset_dir / "archives"
            archives_dir.mkdir(exist_ok=True)

            for archive_file in tqdm(sorted(files["audio_archives"]), desc=f"Downloading {subset} audio"):
                # Download archive
                archive_path = Path(hf_hub_download(
                    BIRDSET_REPO,
                    archive_file,
                    repo_type="dataset",
                    revision=BIRDSET_DATA_BRANCH,
                    local_dir=subset_dir,
                ))

                # Extract to audio directory
                _extract_tar_gz(archive_path, audio_dir, logger)

                # Remove archive after extraction unless keeping
                if not keep_archives:
                    try:
                        archive_path.unlink(missing_ok=True)
                    except OSError:
                        pass  # File may be managed by HF cache

            # Clean up archives directory if empty
            if not keep_archives and archives_dir.exists():
                try:
                    archives_dir.rmdir()
                except OSError:
                    pass  # Directory not empty, leave it

        # Count downloaded files
        audio_count = len(list((subset_dir / "audio").glob("*"))) if (subset_dir / "audio").exists() else 0
        # Parquet files are in nested subdirectory due to hf_hub_download path structure
        meta_count = len(list(subset_dir.glob("**/*.parquet")))

        logger.info(f"Successfully downloaded '{subset}': {meta_count} metadata files, {audio_count} audio files")
        return True

    except Exception as e:
        logger.error(f"Failed to download '{subset}': {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False


def verify_subset(
    subset: str,
    output_dir: Path,
    logger: logging.Logger,
) -> bool:
    """
    Verify a downloaded subset.

    Args:
        subset: Name of the subset to verify
        output_dir: Directory where dataset is saved
        logger: Logger instance

    Returns:
        True if verification passes, False otherwise
    """
    subset_dir = output_dir / subset
    logger.info(f"Verifying subset '{subset}'...")

    try:
        # Check directory exists
        if not subset_dir.exists():
            logger.error(f"Subset directory does not exist: {subset_dir}")
            return False

        # Check for metadata parquet files (hf_hub_download creates nested paths)
        parquet_files = list(subset_dir.glob("**/*.parquet"))

        if not parquet_files:
            logger.error(f"No parquet metadata files found in {subset_dir}")
            return False

        # Check for audio directory
        audio_dir = subset_dir / "audio"
        if not audio_dir.exists():
            logger.error(f"Audio directory does not exist: {audio_dir}")
            return False

        audio_files = list(audio_dir.glob("*"))
        if not audio_files:
            logger.error(f"No audio files found in {audio_dir}")
            return False

        # Try loading metadata to verify integrity
        total_samples = 0
        for pq_file in parquet_files:
            try:
                df = pd.read_parquet(pq_file)
                num_samples = len(df)
                total_samples += num_samples
                logger.debug(f"  {subset}/{pq_file.name}: {num_samples} samples")
            except Exception as e:
                logger.error(f"Failed to read parquet file {pq_file}: {e}")
                return False

        if total_samples == 0:
            logger.warning(f"Subset '{subset}' has no samples in metadata")
            return False

        logger.info(f"Verification passed for '{subset}' ({total_samples} metadata rows, {len(audio_files)} audio files)")
        return True

    except Exception as e:
        logger.error(f"Verification failed for '{subset}': {e}")
        return False


# ============================================================================
# Main Download Function
# ============================================================================


def download_birdset(
    output_dir: Path,
    subsets: Optional[list] = None,
    skip_verification: bool = False,
    force_redownload: bool = False,
    num_proc: int = 4,
    verbose: bool = False,
    keep_archives: bool = False,
) -> dict:
    """
    Download BirdSet dataset with resume capability.

    Args:
        output_dir: Directory to save datasets
        subsets: List of subsets to download (None for all)
        skip_verification: Skip verification after download
        force_redownload: Redownload even if already completed
        num_proc: Number of processes for parallel download (kept for API compat)
        verbose: Enable verbose logging
        keep_archives: Keep downloaded tar.gz archives after extraction

    Returns:
        Dictionary with download results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir, verbose)
    tracker = ProgressTracker(output_dir)

    logger.info("=" * 60)
    logger.info("BirdSet Dataset Downloader")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"HuggingFace cache: {_HF_CACHE_DIR}")

    # Get available subsets
    available = get_available_subsets(logger)

    # Determine which subsets to download
    if subsets is None:
        subsets = [s for s in BIRDSET_SUBSETS if s in available]
    else:
        # Validate requested subsets
        invalid = [s for s in subsets if s not in available]
        if invalid:
            logger.warning(f"Invalid subsets (not available): {invalid}")
            subsets = [s for s in subsets if s in available]

    logger.info(f"Subsets to download: {subsets}")

    # Check for already completed downloads
    if not force_redownload:
        pending = tracker.get_pending_subsets(subsets)
        if len(pending) < len(subsets):
            completed = [s for s in subsets if s not in pending]
            logger.info(f"Already downloaded: {completed}")
            subsets = pending

    if not subsets:
        logger.info("All subsets already downloaded!")
        return {"completed": tracker.progress["completed_subsets"], "failed": {}, "verified": tracker.progress["verified"]}

    logger.info(f"Downloading {len(subsets)} subset(s)...")

    # Download each subset
    results = {"completed": [], "failed": {}, "verified": []}

    for i, subset in enumerate(subsets, 1):
        logger.info("-" * 40)
        logger.info(f"[{i}/{len(subsets)}] Processing subset: {subset}")

        # Download
        success = download_subset(subset, output_dir, logger, num_proc, keep_archives)

        if success:
            tracker.mark_completed(subset)
            results["completed"].append(subset)

            # Verify if requested
            if not skip_verification:
                verified = verify_subset(subset, output_dir, logger)
                if verified:
                    tracker.mark_verified(subset)
                    results["verified"].append(subset)
                else:
                    logger.warning(f"Verification failed for '{subset}'")
        else:
            error_msg = f"Download failed for {subset}"
            tracker.mark_failed(subset, error_msg)
            results["failed"][subset] = error_msg

    # Final summary
    logger.info("=" * 60)
    logger.info("Download Summary")
    logger.info("=" * 60)
    logger.info(f"Completed: {len(results['completed'])} subsets")
    logger.info(f"Failed: {len(results['failed'])} subsets")
    if not skip_verification:
        logger.info(f"Verified: {len(results['verified'])} subsets")

    if results["failed"]:
        logger.warning("Failed subsets:")
        for subset, error in results["failed"].items():
            logger.warning(f"  - {subset}: {error}")

    # Overall progress
    overall = tracker.get_summary()
    logger.info(f"\nOverall progress: {overall['completed']}/{len(BIRDSET_SUBSETS)} subsets downloaded")

    return results


# ============================================================================
# CLI
# ============================================================================


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download BirdSet dataset from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all subsets
  python download_birdset.py

  # Download specific subsets
  python download_birdset.py --subsets HSN NBP POW

  # Force re-download with verbose output
  python download_birdset.py --force --verbose

  # Download to custom directory
  python download_birdset.py --output /path/to/data

  # Skip verification (faster, but less safe)
  python download_birdset.py --skip-verify

Available subsets:
  HSN  - High Sierra Nevada
  NBP  - NorthEast Birds Prediction
  POW  - Powdermill Nature Reserve
  PER  - Peru
  NES  - Northeastern USA/Southeast Canada
  UHH  - University of Hawaii at Hilo
  SSW  - South Sweden
  SNE  - Sierra Nevada East
  XCM  - Xeno-Canto Medium (pretraining)
  XCL  - Xeno-Canto Large (pretraining)
        """,
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )

    parser.add_argument(
        "--subsets",
        "-s",
        nargs="+",
        choices=BIRDSET_SUBSETS,
        default=None,
        help="Specific subsets to download (default: all)",
    )

    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip verification after download",
    )

    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-download even if already completed",
    )

    parser.add_argument(
        "--num-proc",
        "-j",
        type=int,
        default=4,
        help="Number of processes for parallel download (default: 4)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep downloaded tar.gz archives after extraction",
    )

    parser.add_argument(
        "--list-subsets",
        action="store_true",
        help="List available subsets and exit",
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help="Show download status and exit",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # List subsets mode
    if args.list_subsets:
        print("Available BirdSet subsets:")
        print("-" * 40)
        for subset in BIRDSET_SUBSETS:
            print(f"  {subset}")
        print(f"\nTotal: {len(BIRDSET_SUBSETS)} subsets")
        print(f"\nRepository: https://huggingface.co/datasets/{BIRDSET_REPO}")
        return 0

    # Status mode
    if args.status:
        tracker = ProgressTracker(args.output)
        summary = tracker.get_summary()
        print("BirdSet Download Status")
        print("-" * 40)
        print(f"Output directory: {args.output}")
        print(f"Completed: {summary['completed']}/{len(BIRDSET_SUBSETS)} subsets")
        print(f"Verified: {summary['verified']}/{len(BIRDSET_SUBSETS)} subsets")
        print(f"Failed: {summary['failed']} subsets")

        if tracker.progress["completed_subsets"]:
            print(f"\nCompleted subsets: {', '.join(tracker.progress['completed_subsets'])}")
        if tracker.progress["failed_subsets"]:
            print(f"\nFailed subsets:")
            for subset, info in tracker.progress["failed_subsets"].items():
                print(f"  - {subset}: {info['error']}")

        pending = [s for s in BIRDSET_SUBSETS if s not in tracker.progress["completed_subsets"]]
        if pending:
            print(f"\nPending subsets: {', '.join(pending)}")

        return 0

    # Download mode
    try:
        results = download_birdset(
            output_dir=args.output,
            subsets=args.subsets,
            skip_verification=args.skip_verify,
            force_redownload=args.force,
            num_proc=args.num_proc,
            verbose=args.verbose,
            keep_archives=args.keep_archives,
        )

        # Exit with error code if any downloads failed
        if results["failed"]:
            return 1
        return 0

    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
        print("Run again to resume from where you left off.")
        return 130

    except Exception as e:
        print(f"\nFatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
