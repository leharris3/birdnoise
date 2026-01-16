# BirdNoise Model Training and Evaluation Guide

This package contains all essential code and the precomputed prior cache file needed to reproduce training and evaluation.

## What's Included

- ✅ Training scripts (`train_weighted_fusion.py`, `evaluate_models.py`)
- ✅ NatureLM-audio code
- ✅ Prior cache file: `Data\cbi\priors_cache_temp_1766584745.h5` (precomputed eBird priors)
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
   - The prior cache file `Data\cbi\priors_cache_temp_1766584745.h5` should be placed relative to your project root
   - If you extracted the zip to `BirdNoise/`, the cache should be at `BirdNoise/Data\cbi\priors_cache_temp_1766584745.h5`
   - Make sure the cache file exists and is readable

---

## Step 3: Train Stage A (Scalar Weight Model)

Stage A trains a **fixed scalar weight** `w` for fusing audio logits with space-time prior.

### Training Command

```bash
cd Scripts

python train_weighted_fusion.py \
    --stage A \
    --data_dir ../Data/cbi \
    --priors_cache ../Data\cbi\priors_cache_temp_1766584745.h5 \
    --save_dir ../checkpoints_fusion_stage_a \
    --epochs 30 \
    --batch_size 64 \
    --lr 0.001 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --val_split 0.1 \
    --pooling mean \
    --device cuda \
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

python train_weighted_fusion.py \
    --stage B \
    --data_dir ../Data/cbi \
    --priors_cache ../Data\cbi\priors_cache_temp_1766584745.h5 \
    --save_dir ../checkpoints_fusion \
    --epochs 30 \
    --batch_size 96 \
    --lr 0.001 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --val_split 0.1 \
    --pooling mean \
    --w_max 2.0 \
    --gate_hidden_dim 64 \
    --device cuda \
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
python train_weighted_fusion.py \
    --stage B \
    --resume ../checkpoints_fusion_stage_a/checkpoint_epoch30.pth \
    --save_dir ../checkpoints_fusion \
    --epochs 30 \
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

python evaluate_models.py \
    --data_dir ../Data/cbi \
    --priors_cache ../Data\cbi\priors_cache_temp_1766584745.h5 \
    --checkpoint_stage_a ../checkpoints_fusion_stage_a/checkpoint_epoch30.pth \
    --checkpoint_stage_b ../checkpoints_fusion/best_model.pth \
    --output_dir ../evaluation_results \
    --batch_size 64 \
    --device cuda \
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
- Verify the cache file path is correct: `Data\cbi\priors_cache_temp_1766584745.h5`
- Make sure you extracted the zip file completely
- Check that the file exists: `ls -lh Data\cbi\priors_cache_temp_1766584745.h5`

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
- Verify audio directory structure: `Data/cbi/train_audio/{ebird_code}/*.mp3`
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
python train_weighted_fusion.py --stage A --epochs 30 \
    --save_dir ../checkpoints_fusion_stage_a \
    --data_dir ../Data/cbi \
    --priors_cache ../Data\cbi\priors_cache_temp_1766584745.h5

# 4. Train Stage B
python train_weighted_fusion.py --stage B --epochs 30 \
    --save_dir ../checkpoints_fusion \
    --data_dir ../Data/cbi \
    --priors_cache ../Data\cbi\priors_cache_temp_1766584745.h5

# 5. Evaluate all models
python evaluate_models.py \
    --checkpoint_stage_a ../checkpoints_fusion_stage_a/checkpoint_epoch30.pth \
    --checkpoint_stage_b ../checkpoints_fusion/best_model.pth \
    --data_dir ../Data/cbi \
    --priors_cache ../Data\cbi\priors_cache_temp_1766584745.h5
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

