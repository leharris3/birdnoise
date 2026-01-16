"""
Recovery script: If precompute_priors.py computed the data but failed to save,
this script can help recover it by saving to a different location or retrying.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import time

sys.path.insert(0, str(Path(__file__).parent))

from train_weighted_fusion import EBirdPriorWrapper

# Paths
data_dir = Path("../Data/cbi")
priors_dir = Path("../Data/priors")
train_csv = data_dir / "train.csv"
output_path = Path("../Data/cbi/priors_cache.h5")

# Alternative output path if main one fails
alt_output_path = Path("../Data/cbi/priors_cache_alt.h5")

print("=" * 80)
print("PRIOR CACHE RECOVERY SCRIPT")
print("=" * 80)
print("\nThis script will recompute and save the prior cache.")
print("If the main location fails, it will try an alternative location.")
print("=" * 80)

# Load dataset
print(f"\nLoading dataset from {train_csv}...")
df = pd.read_csv(train_csv)
all_species = sorted(df["species"].unique())
num_species = len(all_species)
num_samples = len(df)

print(f"  Samples: {num_samples}")
print(f"  Species: {num_species}")

# Initialize prior model
print(f"\nInitializing prior model...")
prior_model = EBirdPriorWrapper(
    priors_dir=priors_dir,
    species_list=all_species,
    cbi_metadata_path=train_csv,
)

# Pre-allocate matrix
print(f"\nAllocating memory for {num_samples} x {num_species} matrix...")
prior_matrix = np.zeros((num_samples, num_species), dtype=np.float32)
print(f"  Memory: {prior_matrix.nbytes / 1e6:.1f} MB")

# Process in batches (smaller batches to avoid memory issues)
batch_size = 500
print(f"\nComputing priors in batches of {batch_size}...")
print("  (This will take a while - same as full recomputation)")

from tqdm import tqdm

for i in tqdm(range(0, num_samples, batch_size), desc="Processing batches"):
    end_idx = min(i + batch_size, num_samples)
    batch_df = df.iloc[i:end_idx]
    
    latitudes = []
    longitudes = []
    dates = []
    
    for _, row in batch_df.iterrows():
        try:
            lat = float(row.get("latitude", 0.0)) if pd.notna(row.get("latitude")) else 0.0
            lon = float(row.get("longitude", 0.0)) if pd.notna(row.get("longitude")) else 0.0
        except (ValueError, TypeError):
            lat = 0.0
            lon = 0.0
        
        latitudes.append(lat)
        longitudes.append(lon)
        dates.append(row.get("date", ""))
    
    # Get priors for batch
    prior_probs_batch = prior_model.get_prior_probs_batch(latitudes, longitudes, dates)
    
    # Fill matrix
    for j, row_idx in enumerate(range(i, end_idx)):
        prior_matrix[row_idx] = prior_probs_batch[j]

print("\n✓ Computation complete!")

# Try to save
print("\n" + "=" * 80)
print("SAVING CACHE")
print("=" * 80)

# Try main location first
save_success = False
for attempt, path in enumerate([output_path, alt_output_path], 1):
    print(f"\nAttempt {attempt}: Saving to {path}")
    
    # Use temp file approach
    temp_path = path.parent / f"{path.stem}_temp_{int(time.time())}{path.suffix}"
    
    try:
        # Remove existing file if it exists
        if path.exists():
            try:
                path.unlink()
                print(f"  Removed existing file")
            except Exception as e:
                print(f"  Warning: Could not remove existing: {e}")
        
        # Save to temp file
        print(f"  Writing to temporary file: {temp_path.name}")
        with h5py.File(temp_path, 'w', libver='latest', swmr=False) as f:
            f.create_dataset('priors', data=prior_matrix, compression='gzip', compression_opts=4)
            f.create_dataset('species_list', data=[s.encode('utf-8') for s in all_species])
            f.create_dataset('sample_indices', data=df.index.values)
            
            f.attrs['num_samples'] = num_samples
            f.attrs['num_species'] = num_species
            f.attrs['data_dir'] = str(data_dir)
        
        # Rename temp to final
        print(f"  Renaming to final file...")
        temp_path.replace(path)
        
        print(f"✓ Successfully saved to {path}")
        print(f"  File size: {path.stat().st_size / 1e6:.1f} MB")
        save_success = True
        break
        
    except PermissionError as e:
        print(f"❌ Permission denied: {e}")
        if attempt < 2:
            print(f"  Trying alternative location...")
        else:
            print(f"\n  Both locations failed!")
            print(f"  Please:")
            print(f"  1. Close any programs that might have the file open")
            print(f"  2. Check file permissions")
            print(f"  3. Try running as administrator")
    except Exception as e:
        print(f"❌ Error: {e}")
        if attempt < 2:
            print(f"  Trying alternative location...")
        else:
            raise

if not save_success:
    print("\n" + "=" * 80)
    print("SAVE FAILED - BUT DATA IS IN MEMORY")
    print("=" * 80)
    print(f"\nThe prior matrix is computed and in memory:")
    print(f"  Shape: {prior_matrix.shape}")
    print(f"  Memory: {prior_matrix.nbytes / 1e6:.1f} MB")
    print(f"\nYou can:")
    print(f"  1. Save manually using Python:")
    print(f"     import h5py, numpy as np")
    print(f"     # Load prior_matrix from this script")
    print(f"     with h5py.File('your_path.h5', 'w') as f:")
    print(f"         f.create_dataset('priors', data=prior_matrix)")
    print(f"  2. Or try a different output path")
    sys.exit(1)

print("\n" + "=" * 80)
print("SUCCESS!")
print("=" * 80)
print(f"Cache saved to: {path}")
print(f"You can now use it in training with:")
print(f"  --priors_cache {path}")

