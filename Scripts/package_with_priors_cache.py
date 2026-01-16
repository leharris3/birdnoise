"""
Package essential files for training reproduction, INCLUDING the HDF5 prior cache file.

This version includes the precomputed prior cache (priors_cache.h5), which means:
- Colleague does NOT need to download eBird TIF files
- Colleague does NOT need to run precompute_priors.py
- Training can proceed directly using the cached priors

Creates a zip file containing:
- Training scripts
- NatureLM-audio code
- Requirements/dependencies
- HDF5 prior cache file
- README with step-by-step instructions
"""

import zipfile
import os
import shutil
from pathlib import Path
from datetime import datetime


def should_include_file(file_path: Path, root_dir: Path, prior_cache_path: Path = None) -> bool:
    """Determine if a file should be included in the package."""
    rel_path = file_path.relative_to(root_dir)
    
    # Always include the prior cache HDF5 file
    if prior_cache_path and file_path.samefile(prior_cache_path):
        return True
    
    # Always exclude
    exclude_patterns = [
        # Data (except the specific prior cache file)
        'Data/train_audio',  # Audio files - too large
        'Data/priors',  # TIF files - not needed if cache is included
        'checkpoints',
        'checkpoints_beans',
        'checkpoints_fusion',
        'checkpoints_test',
        'evaluation_results',
        
        # Logs and cache (except our specific cache file)
        'wandb',
        '__pycache__',
        '.pyc',
        '.pyd',
        '.pyo',
        '.egg-info',
        '*.log',
        '.git',
        '.vscode',
        '.idea',
        
        # Large or unnecessary files
        '*.tar.gz',
        '*.zip',
        '*.mp3',
        '*.wav',
        # Note: We include specific .h5 files (prior cache) but exclude others
        '*.tif',  # Not needed if cache is included
        '*.png',
        '*.jpg',
        '*.jpeg',
        '*.pdf',
        
        # Temporary and OS files
        '.DS_Store',
        'Thumbs.db',
        '*.tmp',
        '*.temp',
        '*.swp',
        '*.swo',
        '*~',
        
        # Test and debug scripts
        'test_*.py',
        'check_*.py',
        'debug_*.py',
        'quick_test_*.py',
        '*.ipynb',
        'profile_*.py',
        
        # Specific files to exclude
        'oovanger0812@ls6.tacc.utexas.edu',
        'hpc_training_files*.zip',
        'BirdNoise.tar.gz',
        
        # Large assets in NatureLM
        'NatureLM-audio/assets/*.mp3',
        'NatureLM-audio/assets/*.wav',
    ]
    
    # Check against exclude patterns
    path_str = str(rel_path)
    for pattern in exclude_patterns:
        if pattern.startswith('*'):
            if path_str.endswith(pattern[1:]):
                return False
        elif pattern.endswith('*'):
            if path_str.startswith(pattern[:-1]):
                return False
        else:
            if pattern in path_str or path_str.startswith(pattern + '/'):
                return False
    
    # Exclude other .h5 files (except our prior cache)
    if file_path.suffix == '.h5':
        if prior_cache_path and file_path.samefile(prior_cache_path):
            return True  # Include our specific cache file
        return False  # Exclude all other .h5 files
    
    # Include Python files, configs, README, requirements
    include_extensions = ['.py', '.yml', '.yaml', '.json', '.txt', '.md', 
                          '.toml', '.lock', '.sh', '.cfg', '.ini', '.csv']  # Added .csv for train.csv
    
    if file_path.suffix in include_extensions:
        return True
    
    # Include directories that might be needed
    if file_path.is_dir():
        dir_name = file_path.name
        if dir_name in ['NatureLM-audio', 'configs', 'Scripts']:
            return True
    
    return False


def should_include_dir(dir_path: Path, root_dir: Path, prior_cache_path: Path = None) -> bool:
    """Determine if a directory should be traversed."""
    rel_path = dir_path.relative_to(root_dir)
    path_str = str(rel_path)
    
    # Don't traverse excluded directories
    exclude_dirs = [
        'checkpoints', 'wandb', '__pycache__', '.git',
        'checkpoints_beans', 'checkpoints_fusion', 'checkpoints_test',
        'evaluation_results', '.vscode', '.idea', 'assets',
        'Data/train_audio',  # Audio files - too large
        'Data/priors',  # TIF files - not needed
    ]
    
    for exclude in exclude_dirs:
        if path_str.startswith(exclude):
            return False
    
    return True


def create_readme(output_path: Path, prior_cache_rel_path: str) -> str:
    """Create a README file with step-by-step instructions."""
    readme_content = f"""# BirdNoise Model Training and Evaluation Guide

This package contains all essential code and the precomputed prior cache file needed to reproduce training and evaluation.

## What's Included

- ✅ Training scripts (`train_weighted_fusion.py`, `evaluate_models.py`)
- ✅ NatureLM-audio code
- ✅ Prior cache file: `{prior_cache_rel_path}` (precomputed eBird priors)
- ✅ Dependencies (requirements.txt, pyproject.toml)

## What's NOT Included (You Need to Download)

- ❌ CBI dataset (audio files and train.csv)
- ❌ eBird TIF files (NOT needed - priors are precomputed in the cache)

---

## Step 1: Environment Setup

### 1.1 Install Python Dependencies

```bash
# Navigate to the NatureLM-audio directory
cd NatureLM-audio
pip install -r requirements.txt

# Install additional dependencies
pip install scikit-learn seaborn h5py rasterio matplotlib pandas torch soundfile resampy tqdm wandb

# If you have GPU support:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 1.2 Authenticate with HuggingFace

NatureLM-audio requires access to Meta Llama 3.1 8B Instruct:

```bash
# Install HuggingFace CLI if needed
pip install huggingface_hub[cli]

# Login to HuggingFace
huggingface-cli login

# Ensure you have access to Meta Llama 3.1 8B:
# Visit: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
# Click "Request Access" if you haven't already
```

---

## Step 2: Download CBI Dataset

You need to download the CBI (Cornell Bird Identification) dataset:

1. **Download from Kaggle**:
   - Visit: https://www.kaggle.com/c/birdclef-2021/data (or current year's competition)
   - Download the training data

2. **Organize the data**:
   ```
   Data/
   └── cbi/
       ├── train.csv          # Metadata CSV file (MUST include these columns)
       └── train_audio/       # Audio files directory
           ├── abythr1/       # One subdirectory per species (ebird_code)
           │   ├── XC123456.mp3
           │   └── ...
           ├── abitwo/
           └── ...
   ```

3. **Required columns in `train.csv`**:
   - `species`: Full species name (e.g., "American Robin")
   - `ebird_code`: Species code (e.g., "amerob")
   - `filename`: Audio filename (e.g., "XC123456.mp3")
   - `latitude`: Latitude (float)
   - `longitude`: Longitude (float)
   - `date`: Date string (e.g., "2021-05-15")

   **Example row**:
   ```csv
   species,ebird_code,filename,latitude,longitude,date
   "American Robin","amerob","XC123456.mp3",40.5,-74.2,"2021-05-15"
   ```

4. **Verify the prior cache location**:
   - The prior cache file `{prior_cache_rel_path}` should be placed relative to your project root
   - If you extracted the zip to `BirdNoise/`, the cache should be at `BirdNoise/{prior_cache_rel_path}`
   - Make sure the cache file exists and is readable

---

## Step 3: Train Stage A (Scalar Weight Model)

Stage A trains a **fixed scalar weight** `w` for fusing audio logits with space-time prior.

### Training Command

```bash
cd Scripts

python train_weighted_fusion.py \\
    --stage A \\
    --data_dir ../Data/cbi \\
    --priors_cache ../{prior_cache_rel_path} \\
    --save_dir ../checkpoints_fusion_stage_a \\
    --epochs 30 \\
    --batch_size 64 \\
    --lr 0.001 \\
    --weight_decay 0.01 \\
    --warmup_ratio 0.1 \\
    --val_split 0.1 \\
    --pooling mean \\
    --device cuda \\
    --num_workers 4
```

**Note**: We use `--priors_cache` but do NOT provide `--priors_dir` because the priors are precomputed in the cache file.

### What Gets Trained

- ✅ Audio classifier: Linear layer mapping NatureLM features → species logits
- ✅ Temperature parameter: Scalar for temperature scaling
- ✅ Epsilon parameter: Scalar for prior smoothing
- ✅ w_weight: Fixed scalar weight for prior fusion
- ❌ NatureLM encoder: Frozen (not trained)

### Expected Output

- Checkpoints saved every 5 epochs in `../checkpoints_fusion_stage_a/`
- Best model saved as `best_model.pth`
- Final checkpoint at epoch 30: `checkpoint_epoch30.pth`

**Save this checkpoint**: `../checkpoints_fusion_stage_a/checkpoint_epoch30.pth`

---

## Step 4: Train Stage B (Gating Network Model)

Stage B trains a **gating network** `w(a,x,t)` that learns context-dependent weights.

### Training Command

```bash
cd Scripts

python train_weighted_fusion.py \\
    --stage B \\
    --data_dir ../Data/cbi \\
    --priors_cache ../{prior_cache_rel_path} \\
    --save_dir ../checkpoints_fusion \\
    --epochs 30 \\
    --batch_size 96 \\
    --lr 0.001 \\
    --weight_decay 0.01 \\
    --warmup_ratio 0.1 \\
    --val_split 0.1 \\
    --pooling mean \\
    --w_max 2.0 \\
    --gate_hidden_dim 64 \\
    --device cuda \\
    --num_workers 4
```

**Note**: Again, we use `--priors_cache` but no `--priors_dir`.

### What Gets Trained

- ✅ Audio classifier: Linear layer (can initialize from Stage A)
- ✅ Temperature parameter: Scalar
- ✅ Epsilon parameter: Scalar
- ✅ Gate network: MLP that outputs context-dependent weight `w(a,x,t)`
- ❌ NatureLM encoder: Frozen (not trained)

### Optional: Initialize from Stage A

```bash
python train_weighted_fusion.py \\
    --stage B \\
    --resume ../checkpoints_fusion_stage_a/checkpoint_epoch30.pth \\
    --save_dir ../checkpoints_fusion \\
    --epochs 30 \\
    ...  # (same other arguments as above)
```

### Expected Output

- Checkpoints saved every 5 epochs in `../checkpoints_fusion/`
- Best model saved as `best_model.pth`

**Save this checkpoint**: `../checkpoints_fusion/best_model.pth`

---

## Step 5: Evaluate Models on CBI Dataset

Now evaluate all models using `evaluate_models.py`:

- **Prior Model**: eBird prior only
- **Likelihood Model**: NatureLM audio classifier only
- **Posterior Stage A**: Audio + Prior with fixed scalar weight
- **Posterior Stage B**: Audio + Prior with gating network

### Evaluation Command

```bash
cd Scripts

python evaluate_models.py \\
    --data_dir ../Data/cbi \\
    --priors_cache ../{prior_cache_rel_path} \\
    --checkpoint_stage_a ../checkpoints_fusion_stage_a/checkpoint_epoch30.pth \\
    --checkpoint_stage_b ../checkpoints_fusion/best_model.pth \\
    --output_dir ../evaluation_results \\
    --batch_size 64 \\
    --device cuda \\
    --num_workers 4
```

**Note**: We use `--priors_cache` but NOT `--priors_dir`.

### What the Script Does

1. Loads CBI validation/test set
2. Evaluates 4 models (Prior, Likelihood, Posterior A, Posterior B)
3. Computes metrics: Probe (accuracy), R-AUC (retrieval), NMI (clustering)
4. Generates:
   - Comparison table: `comparison_table.png`
   - Confusion matrices: `confusion_matrices.png`
   - Pathological examples: `pathological_example_*.png`

### Expected Output

The script creates:
- `../evaluation_results/comparison_table.png` - Table of metrics
- `../evaluation_results/confusion_matrices.png` - Confusion matrices
- `../evaluation_results/pathological_example_*.png` - Example plots
- `../evaluation_results/results.csv` - Detailed metrics in CSV

---

## Troubleshooting

### Issue: Prior Cache File Not Found

**Error**: `FileNotFoundError: priors_cache.h5`

**Solution**: 
- Verify the cache file path is correct: `{prior_cache_rel_path}`
- Make sure you extracted the zip file completely
- Check that the file exists: `ls -lh {prior_cache_rel_path}`

### Issue: HuggingFace Authentication Errors

**Error**: `OSError: You are trying to access a gated repo...`

**Solution**: 
```bash
huggingface-cli login
# Enter your token from https://huggingface.co/settings/tokens
```

### Issue: CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solution**: Reduce batch size
```bash
--batch_size 32  # or 16 if still too large
```

### Issue: train.csv Missing Columns

**Error**: `KeyError: 'latitude'`

**Solution**: 
- Verify `train.csv` has all required columns: `species`, `ebird_code`, `filename`, `latitude`, `longitude`, `date`
- Check column names match exactly (case-sensitive)

### Issue: Audio Files Not Found

**Error**: `FileNotFoundError: train_audio/...`

**Solution**: 
- Verify audio directory structure: `Data/cbi/train_audio/{{ebird_code}}/*.mp3`
- Check that filenames in `train.csv` match actual audio files
- Ensure audio files are in MP3 format (or modify dataset code to support other formats)

---

## Quick Start (Summary)

```bash
# 1. Setup environment
cd NatureLM-audio
pip install -r requirements.txt
pip install scikit-learn seaborn h5py rasterio
huggingface-cli login

# 2. Download CBI dataset to Data/cbi/ (see Step 2)

# 3. Train Stage A
cd Scripts
python train_weighted_fusion.py --stage A --epochs 30 \\
    --save_dir ../checkpoints_fusion_stage_a \\
    --data_dir ../Data/cbi \\
    --priors_cache ../{prior_cache_rel_path}

# 4. Train Stage B
python train_weighted_fusion.py --stage B --epochs 30 \\
    --save_dir ../checkpoints_fusion \\
    --data_dir ../Data/cbi \\
    --priors_cache ../{prior_cache_rel_path}

# 5. Evaluate all models
python evaluate_models.py \\
    --checkpoint_stage_a ../checkpoints_fusion_stage_a/checkpoint_epoch30.pth \\
    --checkpoint_stage_b ../checkpoints_fusion/best_model.pth \\
    --data_dir ../Data/cbi \\
    --priors_cache ../{prior_cache_rel_path}
```

---

## Expected Training Time

- **Stage A**: ~4-6 hours on GPU (30 epochs, batch_size=64)
- **Stage B**: ~4-6 hours on GPU (30 epochs, batch_size=96)
- **Evaluation**: ~30-60 minutes (single run on all 4 models)

**Total**: ~10-15 hours

---

## File Checklist

After completing all steps, you should have:

**Training outputs**:
- [ ] `checkpoints_fusion_stage_a/checkpoint_epoch30.pth` - Stage A model
- [ ] `checkpoints_fusion_stage_a/best_model.pth` - Stage A best model
- [ ] `checkpoints_fusion/best_model.pth` - Stage B model

**Evaluation outputs**:
- [ ] `evaluation_results/comparison_table.png` - Metrics table
- [ ] `evaluation_results/confusion_matrices.png` - Confusion matrices
- [ ] `evaluation_results/pathological_example_*.png` - Example plots
- [ ] `evaluation_results/results.csv` - Detailed results

---

## Notes

1. **Prior Cache**: The included HDF5 file contains precomputed priors for all samples in the training dataset. You do NOT need to download eBird TIF files or run `precompute_priors.py`.

2. **Data Split**: The training script uses `--val_split 0.1` (10% validation split). The evaluation script evaluates on the validation set.

3. **Reproducibility**: Scripts use `--seed 42` by default for reproducible results.

4. **Monitoring**: Training logs to Weights & Biases (wandb) by default. Disable with `--no_wandb` if needed.

5. **Stage Differences**:
   - **Stage A**: Simple scalar weight `w` learned for all samples
   - **Stage B**: Learned gating network `w(a,x,t)` that adapts weight based on:
     - Audio model confidence
     - Prior confidence
     - Metadata (location, time)

---

For questions or issues, refer to the code comments in the training scripts or contact the original authors.

"""
    return readme_content


def create_package(output_path: Path, root_dir: Path = None, prior_cache_path: str = None):
    """Create a zip package with essential files including the prior cache."""
    if root_dir is None:
        root_dir = Path(__file__).parent.parent  # Project root
    
    # Find prior cache file
    prior_cache_file = None
    if prior_cache_path:
        prior_cache_file = Path(prior_cache_path)
        if not prior_cache_file.is_absolute():
            prior_cache_file = root_dir / prior_cache_file
    else:
        # Try default location
        default_paths = [
            root_dir / "Data" / "cbi" / "priors_cache.h5",
            root_dir / "Data" / "priors_cache.h5",
        ]
        for path in default_paths:
            if path.exists():
                prior_cache_file = path
                break
    
    if prior_cache_file is None or not prior_cache_file.exists():
        raise FileNotFoundError(
            f"Prior cache file not found! Expected at one of:\n"
            f"  - {prior_cache_path} (if provided)\n"
            f"  - {root_dir / 'Data' / 'cbi' / 'priors_cache.h5'}\n"
            f"  - {root_dir / 'Data' / 'priors_cache.h5'}\n"
            f"Please specify --prior_cache_path or ensure the file exists."
        )
    
    output_path = Path(output_path)
    if output_path.suffix != '.zip':
        output_path = output_path.with_suffix('.zip')
    
    print(f"Creating reproduction package: {output_path}")
    print(f"Root directory: {root_dir}")
    print(f"Prior cache file: {prior_cache_file}")
    print(f"Cache file size: {prior_cache_file.stat().st_size / (1024*1024):.2f} MB")
    print()
    
    included_files = []
    excluded_files = []
    
    # Compute relative path for cache file
    prior_cache_rel = prior_cache_file.relative_to(root_dir)
    
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add files
        for root, dirs, files in os.walk(root_dir):
            root_path = Path(root)
            
            # Filter directories to traverse
            dirs[:] = [d for d in dirs if should_include_dir(root_path / d, root_dir, prior_cache_file)]
            
            for file in files:
                file_path = root_path / file
                
                if should_include_file(file_path, root_dir, prior_cache_file):
                    rel_path = file_path.relative_to(root_dir)
                    zipf.write(file_path, rel_path)
                    included_files.append(rel_path)
                else:
                    rel_path = file_path.relative_to(root_dir)
                    excluded_files.append(rel_path)
        
        # Create README file
        readme_content = create_readme(output_path, str(prior_cache_rel))
        zipf.writestr("README.md", readme_content)
        included_files.append(Path("README.md"))
        print(f"Created README.md with instructions")
    
    print(f"\n{'='*60}")
    print(f"Package created: {output_path}")
    print(f"{'='*60}")
    print(f"Included files: {len(included_files)}")
    print(f"Excluded files: {len(excluded_files)}")
    print(f"\nPackage size: {output_path.stat().st_size / (1024*1024):.2f} MB")
    
    print(f"\n{'='*60}")
    print("Key files included:")
    print(f"{'='*60}")
    
    key_files = [f for f in included_files if Path(f).name in [
        'requirements.txt', 'README.md', 'pyproject.toml', 
        'train_weighted_fusion.py', 'train_lc.py', 'evaluate_models.py',
        'eBirdPrior.py', 'precompute_priors.py', 'priors_cache.h5'
    ]]
    
    for key_file in sorted(key_files):
        size = "N/A"
        if key_file == prior_cache_rel:
            size = f"{prior_cache_file.stat().st_size / (1024*1024):.1f} MB"
        print(f"  {key_file} {size}")
    
    print(f"\n{'='*60}")
    print("Next Steps:")
    print(f"{'='*60}")
    print("1. Share the zip file with your colleague")
    print("2. They should extract it and read README.md")
    print("3. They need to download only the CBI dataset (audio files + train.csv)")
    print("4. They do NOT need eBird TIF files (priors are precomputed)")
    print("5. They can proceed directly to training Stage A and B")
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Package essential files for reproduction (WITH prior cache HDF5 file)"
    )
    parser.add_argument("--output", type=str, default=None,
                       help="Output zip file path (default: BirdNoise_with_cache_YYYYMMDD.zip)")
    parser.add_argument("--root", type=str, default=None,
                       help="Root directory to package (default: parent of Scripts)")
    parser.add_argument("--prior_cache_path", type=str, default=None,
                       help="Path to prior cache HDF5 file (default: Data/cbi/priors_cache.h5)")
    
    args = parser.parse_args()
    
    # Default output path
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d")
        script_dir = Path(__file__).parent
        args.output = script_dir.parent / f"BirdNoise_with_cache_{timestamp}.zip"
    
    root_dir = Path(args.root) if args.root else Path(__file__).parent.parent
    
    create_package(args.output, root_dir, args.prior_cache_path)
