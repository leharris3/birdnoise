#!/usr/bin/env python3
"""
BEANS Dataset Downloader
========================
Downloads datasets from the BEANS (Benchmark of Animal Sounds) benchmark,
excluding the CBI (Cornell Bird Identification) sub-dataset which requires
Kaggle competition registration.

Datasets included:
- ESC-50: Environmental Sound Classification
- Watkins: Marine Mammal Sound Database  
- Bats: Egyptian Fruit Bat Vocalizations
- HumBugDB: Mosquito Wingbeat Sounds
- Speech Commands: Google Speech Commands v0.02
- Dogs: Dog bark dataset (manual download instructions provided)

Usage:
    python download_beans_datasets.py [--output-dir OUTPUT_DIR] [--datasets DATASET1,DATASET2,...]

Requirements:
    pip install requests tqdm zenodo_get

Author: Generated for BEANS benchmark research
License: MIT
"""

import os
import sys
import argparse
import subprocess
import zipfile
import tarfile
import hashlib
import shutil
from pathlib import Path
from typing import Optional, List, Dict
import urllib.request
import json

# Try to import optional dependencies
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("Warning: 'requests' not installed. Using urllib instead.")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: 'tqdm' not installed. Progress bars will be disabled.")


# Dataset configurations
DATASETS = {
    "esc50": {
        "name": "ESC-50 (Environmental Sound Classification)",
        "url": "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip",
        "description": "2000 environmental audio recordings, 50 classes, 5 seconds each",
        "size_mb": 600,
        "license": "CC BY-NC 3.0",
        "citation": "Piczak, K. J. (2015). ESC: Dataset for Environmental Sound Classification. ACM MM.",
    },
    "watkins": {
        "name": "Watkins Marine Mammal Sound Database",
        "url": "manual",  # Requires web scraping or manual download
        "website": "https://cis.whoi.edu/science/B/whalesounds/",
        "description": "~15,000 annotated clips of 60+ marine mammal species",
        "size_mb": 5000,  # Approximate
        "license": "Free for personal/academic use",
        "citation": "Sayigh et al. (2016). Watkins Marine Mammal Sound Database.",
        "note": "Manual download required from WHOI website",
    },
    "bats": {
        "name": "Egyptian Fruit Bat Vocalizations",
        "url": "figshare",
        "figshare_id": "3666502",
        "description": "~300,000 bat vocalization files with annotations",
        "size_mb": 50000,  # Very large dataset
        "license": "CC BY 4.0",
        "citation": "Prat et al. (2017). Scientific Data 4:170143.",
    },
    "humbugdb": {
        "name": "HumBugDB (Mosquito Wingbeats)",
        "url": "zenodo",
        "zenodo_id": "4904800",
        "description": "20 hours of mosquito audio, 36 species",
        "size_mb": 2000,
        "license": "CC BY 4.0",
        "citation": "Kiskin et al. (2021). NeurIPS Datasets and Benchmarks.",
    },
    "speech": {
        "name": "Google Speech Commands v0.02",
        "url": "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz",
        "description": "105,829 one-second audio clips of 35 spoken words",
        "size_mb": 2300,
        "license": "CC BY 4.0",
        "citation": "Warden, P. (2018). Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition.",
    },
    "dogs": {
        "name": "Dog Bark Dataset",
        "url": "manual",
        "description": "Barks from 10 dogs in disturbance, isolation, and play contexts",
        "size_mb": 100,  # Approximate
        "license": "Academic use",
        "citation": "Yin & McCowan (2004). Behavioural Processes.",
        "note": "This dataset may require contacting the original authors for access.",
    },
}


class DownloadProgressBar:
    """Progress bar for downloads."""
    
    def __init__(self, total: int, desc: str = "Downloading"):
        self.total = total
        self.desc = desc
        self.current = 0
        if TQDM_AVAILABLE:
            self.pbar = tqdm(total=total, desc=desc, unit='B', unit_scale=True)
        else:
            self.pbar = None
    
    def update(self, block_size: int):
        self.current += block_size
        if self.pbar:
            self.pbar.update(block_size)
        else:
            percent = (self.current / self.total) * 100 if self.total > 0 else 0
            print(f"\r{self.desc}: {percent:.1f}%", end='', flush=True)
    
    def close(self):
        if self.pbar:
            self.pbar.close()
        else:
            print()  # New line


def download_file(url: str, dest_path: Path, desc: str = "Downloading") -> bool:
    """Download a file with progress bar."""
    try:
        if REQUESTS_AVAILABLE:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            
            progress = DownloadProgressBar(total_size, desc)
            with open(dest_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        progress.update(len(chunk))
            progress.close()
        else:
            # Fallback to urllib
            def reporthook(block_num, block_size, total_size):
                if not hasattr(reporthook, 'pbar'):
                    reporthook.pbar = DownloadProgressBar(total_size, desc)
                reporthook.pbar.update(block_size)
            
            urllib.request.urlretrieve(url, dest_path, reporthook)
            if hasattr(reporthook, 'pbar'):
                reporthook.pbar.close()
        
        return True
    except Exception as e:
        print(f"\nError downloading {url}: {e}")
        return False


def extract_archive(archive_path: Path, extract_dir: Path) -> bool:
    """Extract zip or tar.gz archive."""
    try:
        print(f"Extracting {archive_path.name}...")
        
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zf:
                zf.extractall(extract_dir)
        elif archive_path.name.endswith('.tar.gz') or archive_path.name.endswith('.tgz'):
            with tarfile.open(archive_path, 'r:gz') as tf:
                tf.extractall(extract_dir)
        elif archive_path.name.endswith('.tar'):
            with tarfile.open(archive_path, 'r') as tf:
                tf.extractall(extract_dir)
        else:
            print(f"Unknown archive format: {archive_path}")
            return False
        
        return True
    except Exception as e:
        print(f"Error extracting {archive_path}: {e}")
        return False


def download_esc50(output_dir: Path) -> bool:
    """Download ESC-50 dataset."""
    dataset = DATASETS["esc50"]
    dest_dir = output_dir / "esc50"
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = dest_dir / "esc50.zip"
    
    print(f"\n{'='*60}")
    print(f"Downloading: {dataset['name']}")
    print(f"Size: ~{dataset['size_mb']} MB")
    print(f"{'='*60}")
    
    if download_file(dataset['url'], zip_path, "ESC-50"):
        if extract_archive(zip_path, dest_dir):
            zip_path.unlink()  # Remove zip after extraction
            print(f"✓ ESC-50 downloaded to {dest_dir}")
            return True
    
    return False


def download_speech_commands(output_dir: Path) -> bool:
    """Download Google Speech Commands v0.02 dataset."""
    dataset = DATASETS["speech"]
    dest_dir = output_dir / "speech_commands"
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    tar_path = dest_dir / "speech_commands_v0.02.tar.gz"
    
    print(f"\n{'='*60}")
    print(f"Downloading: {dataset['name']}")
    print(f"Size: ~{dataset['size_mb']} MB")
    print(f"{'='*60}")
    
    if download_file(dataset['url'], tar_path, "Speech Commands"):
        if extract_archive(tar_path, dest_dir):
            tar_path.unlink()  # Remove tar after extraction
            print(f"✓ Speech Commands downloaded to {dest_dir}")
            return True
    
    return False


def download_humbugdb(output_dir: Path) -> bool:
    """Download HumBugDB dataset from Zenodo."""
    dataset = DATASETS["humbugdb"]
    dest_dir = output_dir / "humbugdb"
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Downloading: {dataset['name']}")
    print(f"Size: ~{dataset['size_mb']} MB")
    print(f"Zenodo ID: {dataset['zenodo_id']}")
    print(f"{'='*60}")
    
    # Try using zenodo_get if available
    try:
        result = subprocess.run(
            ["zenodo_get", dataset['zenodo_id'], "-o", str(dest_dir)],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"✓ HumBugDB downloaded to {dest_dir}")
            return True
        else:
            print(f"zenodo_get failed: {result.stderr}")
    except FileNotFoundError:
        print("zenodo_get not found. Install with: pip install zenodo_get")
    
    # Fallback: provide manual instructions
    zenodo_url = f"https://zenodo.org/records/{dataset['zenodo_id']}"
    print(f"\nManual download required:")
    print(f"  1. Visit: {zenodo_url}")
    print(f"  2. Download all files to: {dest_dir}")
    print(f"  3. Extract the multi-part zip archive")
    
    # Create a README with instructions
    readme_path = dest_dir / "DOWNLOAD_INSTRUCTIONS.txt"
    with open(readme_path, 'w') as f:
        f.write(f"HumBugDB Download Instructions\n")
        f.write(f"{'='*40}\n\n")
        f.write(f"1. Visit: {zenodo_url}\n")
        f.write(f"2. Download all audio zip files\n")
        f.write(f"3. Extract into this directory\n")
        f.write(f"4. Download the metadata CSV file\n\n")
        f.write(f"Alternative: pip install zenodo_get && zenodo_get {dataset['zenodo_id']}\n")
    
    return False


def download_bats(output_dir: Path) -> bool:
    """Download Egyptian Fruit Bat dataset from Figshare."""
    dataset = DATASETS["bats"]
    dest_dir = output_dir / "bats"
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Downloading: {dataset['name']}")
    print(f"Size: ~{dataset['size_mb']} MB (VERY LARGE)")
    print(f"Figshare Collection ID: {dataset['figshare_id']}")
    print(f"{'='*60}")
    
    figshare_url = f"https://figshare.com/collections/An_annotated_dataset_of_Egyptian_fruit_bat_vocalizations_across_varying_contexts_and_during_vocal_ontogeny/{dataset['figshare_id']}/2"
    
    print(f"\n⚠️  This is a very large dataset (~50 GB)")
    print(f"Manual download recommended:")
    print(f"  1. Visit: {figshare_url}")
    print(f"  2. Download individual files as needed")
    print(f"  3. Save to: {dest_dir}")
    
    # Try to get the collection metadata
    api_url = f"https://api.figshare.com/v2/collections/{dataset['figshare_id']}/articles"
    try:
        if REQUESTS_AVAILABLE:
            response = requests.get(api_url, timeout=30)
            if response.status_code == 200:
                articles = response.json()
                print(f"\nFound {len(articles)} items in collection:")
                for article in articles[:5]:  # Show first 5
                    print(f"  - {article.get('title', 'Unknown')}")
                if len(articles) > 5:
                    print(f"  ... and {len(articles) - 5} more")
    except Exception as e:
        print(f"Could not fetch collection metadata: {e}")
    
    # Create instructions file
    readme_path = dest_dir / "DOWNLOAD_INSTRUCTIONS.txt"
    with open(readme_path, 'w') as f:
        f.write(f"Egyptian Fruit Bat Dataset Download Instructions\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"This is a large dataset (~50 GB) hosted on Figshare.\n\n")
        f.write(f"Collection URL: {figshare_url}\n")
        f.write(f"DOI: https://doi.org/10.6084/m9.figshare.c.{dataset['figshare_id']}\n\n")
        f.write(f"Download the files you need and place them in this directory.\n")
        f.write(f"\nCitation:\n{dataset['citation']}\n")
    
    return False


def download_watkins(output_dir: Path) -> bool:
    """Provide instructions for Watkins Marine Mammal Sound Database."""
    dataset = DATASETS["watkins"]
    dest_dir = output_dir / "watkins"
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset['name']}")
    print(f"{'='*60}")
    
    print(f"\n⚠️  Manual download required")
    print(f"The Watkins database requires downloading from WHOI website.")
    print(f"\nInstructions:")
    print(f"  1. Visit: {dataset['website']}")
    print(f"  2. Navigate to 'All cuts' or 'Best of cuts'")
    print(f"  3. Download the species you need")
    print(f"  4. Save files to: {dest_dir}")
    
    # Create instructions file
    readme_path = dest_dir / "DOWNLOAD_INSTRUCTIONS.txt"
    with open(readme_path, 'w') as f:
        f.write(f"Watkins Marine Mammal Sound Database\n")
        f.write(f"{'='*40}\n\n")
        f.write(f"Website: {dataset['website']}\n\n")
        f.write(f"This database contains recordings of 60+ marine mammal species.\n")
        f.write(f"Files are free for personal and academic use.\n\n")
        f.write(f"Download options:\n")
        f.write(f"  - 'Best of' cuts: Higher quality, curated selection\n")
        f.write(f"  - 'All cuts': Complete collection (~15,000 clips)\n")
        f.write(f"  - 'Master tapes': Full original recordings\n\n")
        f.write(f"License: {dataset['license']}\n")
        f.write(f"Citation: {dataset['citation']}\n")
    
    print(f"\nInstructions saved to: {readme_path}")
    return False


def download_dogs(output_dir: Path) -> bool:
    """Provide instructions for Dogs bark dataset."""
    dataset = DATASETS["dogs"]
    dest_dir = output_dir / "dogs"
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset['name']}")
    print(f"{'='*60}")
    
    print(f"\n⚠️  Manual download required")
    print(f"This dataset may require contacting the original authors.")
    print(f"\nCitation: {dataset['citation']}")
    print(f"\nThe dataset contains barks from 10 domestic dogs in three contexts:")
    print(f"  - Disturbance (stranger approaching)")
    print(f"  - Isolation (dog left alone)")
    print(f"  - Play (playing with owner/other dogs)")
    
    # Create instructions file
    readme_path = dest_dir / "DOWNLOAD_INSTRUCTIONS.txt"
    with open(readme_path, 'w') as f:
        f.write(f"Dog Bark Dataset\n")
        f.write(f"{'='*40}\n\n")
        f.write(f"Description: {dataset['description']}\n\n")
        f.write(f"This dataset was used in the BEANS benchmark for individual\n")
        f.write(f"dog identification from bark sounds.\n\n")
        f.write(f"Original Paper:\n")
        f.write(f"  Yin, S., & McCowan, B. (2004). Barking in domestic dogs:\n")
        f.write(f"  context specificity and individual identification.\n")
        f.write(f"  Animal Behaviour, 68(2), 343-355.\n\n")
        f.write(f"To obtain the dataset:\n")
        f.write(f"  1. Contact the authors through their institution\n")
        f.write(f"  2. Check if the BEANS repository has a download script\n")
        f.write(f"  3. Search for alternative dog bark datasets\n\n")
        f.write(f"Alternative datasets:\n")
        f.write(f"  - UrbanSound8K (contains dog_bark class)\n")
        f.write(f"  - ESC-50 (contains dog class)\n")
    
    print(f"\nInstructions saved to: {readme_path}")
    return False


def create_summary(output_dir: Path, results: Dict[str, bool]):
    """Create a summary file of download results."""
    summary_path = output_dir / "DOWNLOAD_SUMMARY.txt"
    
    with open(summary_path, 'w') as f:
        f.write("BEANS Dataset Download Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Output directory: {output_dir}\n\n")
        
        f.write("Download Results:\n")
        f.write("-" * 30 + "\n")
        for dataset, success in results.items():
            status = "✓ Downloaded" if success else "⚠ Manual download required"
            f.write(f"  {dataset}: {status}\n")
        
        f.write("\n\nDataset Details:\n")
        f.write("-" * 30 + "\n")
        for name, info in DATASETS.items():
            if name == "cbi":
                continue
            f.write(f"\n{info['name']}\n")
            f.write(f"  Description: {info['description']}\n")
            f.write(f"  License: {info['license']}\n")
            f.write(f"  Citation: {info['citation']}\n")
    
    print(f"\nSummary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download BEANS benchmark datasets (excluding CBI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_beans_datasets.py
  python download_beans_datasets.py --output-dir ./data
  python download_beans_datasets.py --datasets esc50,speech
  python download_beans_datasets.py --list

Available datasets: esc50, watkins, bats, humbugdb, speech, dogs
        """
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("./beans_data"),
        help="Output directory for downloaded datasets (default: ./beans_data)"
    )
    
    parser.add_argument(
        "--datasets", "-d",
        type=str,
        default="all",
        help="Comma-separated list of datasets to download, or 'all' (default: all)"
    )
    
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available datasets and exit"
    )
    
    args = parser.parse_args()
    
    # List datasets
    if args.list:
        print("\nAvailable BEANS Datasets (excluding CBI):")
        print("=" * 60)
        for name, info in DATASETS.items():
            auto = "Auto" if info['url'] not in ["manual", "figshare", "zenodo"] else "Manual"
            print(f"\n  {name}")
            print(f"    Name: {info['name']}")
            print(f"    Size: ~{info['size_mb']} MB")
            print(f"    Download: {auto}")
        return
    
    # Determine which datasets to download
    if args.datasets.lower() == "all":
        to_download = list(DATASETS.keys())
    else:
        to_download = [d.strip().lower() for d in args.datasets.split(",")]
        invalid = [d for d in to_download if d not in DATASETS]
        if invalid:
            print(f"Error: Unknown datasets: {invalid}")
            print(f"Available: {list(DATASETS.keys())}")
            return
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("BEANS Dataset Downloader")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Datasets to download: {to_download}")
    
    # Download functions mapping
    download_funcs = {
        "esc50": download_esc50,
        "watkins": download_watkins,
        "bats": download_bats,
        "humbugdb": download_humbugdb,
        "speech": download_speech_commands,
        "dogs": download_dogs,
    }
    
    results = {}
    
    for dataset in to_download:
        if dataset in download_funcs:
            results[dataset] = download_funcs[dataset](args.output_dir)
        else:
            print(f"No download function for: {dataset}")
            results[dataset] = False
    
    # Create summary
    create_summary(args.output_dir, results)
    
    # Final status
    print("\n" + "=" * 60)
    print("Download Complete!")
    print("=" * 60)
    
    auto_downloaded = [k for k, v in results.items() if v]
    manual_required = [k for k, v in results.items() if not v]
    
    if auto_downloaded:
        print(f"\n✓ Successfully downloaded: {', '.join(auto_downloaded)}")
    
    if manual_required:
        print(f"\n⚠ Manual download required for: {', '.join(manual_required)}")
        print("  Check the DOWNLOAD_INSTRUCTIONS.txt files in each dataset folder.")


if __name__ == "__main__":
    main()
