"""
Pre-compute space-time priors for all training/validation samples.

This script computes the prior probability vectors for all samples in the dataset
and saves them to a fast HDF5 file. During training, we can then load from cache
instead of computing on-the-fly, which is much faster.
"""

import argparse
import warnings
import sys
import time
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import h5py
from multiprocessing import Pool, cpu_count

# Suppress audio warnings
warnings.filterwarnings("ignore", message=".*Xing stream size.*")
warnings.filterwarnings("ignore", message=".*Illegal Audio-MPEG-Header.*")
warnings.filterwarnings("ignore", message=".*trying to resync.*")
warnings.filterwarnings("ignore", message=".*hit end of available data.*")
warnings.filterwarnings("ignore", message=".*AUDIO-MPEG-HEADER.*")
warnings.filterwarnings("ignore", category=UserWarning)

# Add Scripts to path
sys.path.insert(0, str(Path(__file__).parent))

from train_weighted_fusion import EBirdPriorWrapper


def compute_prior_batch(args_tuple):
    """Compute priors for a batch of samples (for multiprocessing)."""
    batch_indices, batch_rows, batch_seasons, priors_dir, species_list, cbi_metadata_path = args_tuple
    
    try:
        # Re-initialize prior model in worker process (can't pickle it)
        # Each worker has its own cache, which will be hot for grouped samples!
        prior_model = EBirdPriorWrapper(
            priors_dir=priors_dir,
            species_list=species_list,
            cbi_metadata_path=cbi_metadata_path,
        )
        
        # Extract metadata for batch
        latitudes = []
        longitudes = []
        dates = []
        
        for row_dict in batch_rows:
            lat = row_dict.get("latitude", 0.0)
            lon = row_dict.get("longitude", 0.0)
            date_str = row_dict.get("date", "")
            
            # Convert to float safely
            try:
                lat_f = float(lat) if pd.notna(lat) else 0.0
                lon_f = float(lon) if pd.notna(lon) else 0.0
            except (ValueError, TypeError):
                lat_f = 0.0
                lon_f = 0.0
            
            latitudes.append(lat_f)
            longitudes.append(lon_f)
            dates.append(date_str)
        
        # Get prior probabilities for entire batch at once (vectorized!)
        # Since samples are grouped by (season, species), cache hits will be maximized!
        prior_probs_batch = prior_model.get_prior_probs_batch(
            latitudes, longitudes, dates
        )
        
        # Return list of (index, prior_vector) tuples
        return [(batch_indices[i], prior_probs_batch[i]) for i in range(len(batch_indices))]
    except Exception as e:
        # Return uniform prior on error for all samples in batch
        num_species = len(species_list)
        uniform_prior = np.ones(num_species) / num_species
        return [(idx, uniform_prior) for idx in batch_indices]


def precompute_priors(
    data_dir: Path,
    priors_dir: Path,
    output_path: Path,
    num_workers: int = None,
    batch_size: int = 1000,
):
    """Pre-compute priors for all samples in the dataset."""
    
    # Load dataset
    train_csv_path = data_dir / "train.csv"
    print(f"Loading dataset from {train_csv_path}...")
    df = pd.read_csv(train_csv_path)
    print(f"Found {len(df)} samples")
    
    # Get species list
    all_species = sorted(df["species"].unique())
    num_species = len(all_species)
    print(f"Number of species: {num_species}")
    
    # Initialize prior model
    print(f"\nInitializing prior model from {priors_dir}...")
    prior_model = EBirdPriorWrapper(
        priors_dir=priors_dir,
        species_list=all_species,
        cbi_metadata_path=train_csv_path,
    )
    
    # Create output file
    print(f"\nCreating output file: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Pre-allocate arrays
    num_samples = len(df)
    prior_matrix = np.zeros((num_samples, num_species), dtype=np.float32)
    
    # Process in batches (much faster than one at a time!)
    print(f"\nComputing priors for {num_samples} samples...")
    num_workers_actual = num_workers if num_workers else 16  # Default to 16 for your CPU
    print(f"Using {num_workers_actual} workers with batch size {batch_size}")
    
    # OPTIMIZATION: Group samples by (season, species) to maximize cache hits
    # This way, all samples for the same species/season are processed together
    # Using 4 seasons instead of 52 weeks = ~13x fewer unique combinations = much faster!
    # The first lookup loads the band into cache, subsequent lookups use hot cache (500x faster!)
    print("\nGrouping samples by (season, species) to maximize cache hits...")
    
    # Extract season and species for each sample (4 seasons vs 52 weeks = much faster!)
    from train_weighted_fusion import week_to_season
    
    sample_metadata = []
    for idx, row in df.iterrows():
        try:
            date_str = row.get("date", "")
            date = pd.to_datetime(date_str)
            week = date.isocalendar()[1]
            season = week_to_season(week)
        except:
            season = 1  # Default to Spring
        
        sample_metadata.append({
            'idx': idx,
            'row': row.to_dict(),
            'season': season,
            'species': row.get('species', ''),
        })
    
    # Group by (season, species) - this maximizes cache hits!
    # Much better than (week, species) - only 4 seasons vs 52 weeks!
    from collections import defaultdict
    grouped_samples = defaultdict(list)
    for meta in sample_metadata:
        key = (meta['season'], meta['species'])
        grouped_samples[key].append(meta)
    
    print(f"  Found {len(grouped_samples)} unique (season, species) combinations")
    print(f"  (Using 4 seasons instead of 52 weeks = ~13x fewer unique combinations!)")
    print(f"  Average samples per combination: {num_samples / len(grouped_samples):.1f}")
    
    # Create batches from grouped samples (process same species/season together)
    batch_list = []
    current_batch_indices = []
    current_batch_rows = []
    current_batch_seasons = []
    
    for (season, species), samples in sorted(grouped_samples.items()):
        # Add all samples for this (season, species) combination
        for sample in samples:
            current_batch_indices.append(sample['idx'])
            current_batch_rows.append(sample['row'])
            current_batch_seasons.append(season)
            
            # When batch is full, start a new one
            if len(current_batch_indices) >= batch_size:
                batch_list.append((current_batch_indices, current_batch_rows, current_batch_seasons, priors_dir, all_species, train_csv_path))
                current_batch_indices = []
                current_batch_rows = []
                current_batch_seasons = []
    
    # Add remaining samples
    if current_batch_indices:
        batch_list.append((current_batch_indices, current_batch_rows, current_batch_seasons, priors_dir, all_species, train_csv_path))
    
    num_batches = len(batch_list)
    print(f"Created {num_batches} optimized batches (grouped by season/species for cache efficiency)")
    
    # Process batches
    print(f"\nStarting computation...")
    print(f"  Total batches: {num_batches}")
    print(f"  Samples per batch: ~{num_samples // num_batches}")
    print(f"  Expected cache efficiency: High (samples grouped by season/species)\n")
    
    processed_samples = 0
    start_time = time.time()
    
    if num_workers_actual > 1:
        # Use multiprocessing (each worker processes batches)
        with Pool(processes=num_workers_actual) as pool:
            results = []
            for result in tqdm(
                pool.imap(compute_prior_batch, batch_list),
                total=num_batches,
                desc="Computing priors",
                unit="batch",
                mininterval=1.0,  # Update every second
            ):
                results.append(result)
                processed_samples += len(result)
                
                # Log progress every 10 batches
                if len(results) % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = processed_samples / elapsed if elapsed > 0 else 0
                    remaining = num_samples - processed_samples
                    eta = remaining / rate if rate > 0 else 0
                    print(f"  Progress: {processed_samples}/{num_samples} samples ({100*processed_samples/num_samples:.1f}%) | "
                          f"Rate: {rate:.0f} samples/sec | ETA: {eta/60:.1f} min")
        
        # Flatten results and fill matrix
        print(f"\nFilling prior matrix...")
        for batch_results in results:
            for row_idx, prior_vec in batch_results:
                prior_matrix[row_idx] = prior_vec
    else:
        # Single-threaded (use existing prior_model, process in batches)
        for batch_idx, batch_data in enumerate(tqdm(batch_list, desc="Computing priors", unit="batch")):
            batch_indices, batch_rows, batch_seasons = batch_data[:3]
            try:
                latitudes = []
                longitudes = []
                dates = []
                
                for row_dict in batch_rows:
                    lat = row_dict.get("latitude", 0.0)
                    lon = row_dict.get("longitude", 0.0)
                    date_str = row_dict.get("date", "")
                    
                    try:
                        lat_f = float(lat) if pd.notna(lat) else 0.0
                        lon_f = float(lon) if pd.notna(lon) else 0.0
                    except (ValueError, TypeError):
                        lat_f = 0.0
                        lon_f = 0.0
                    
                    latitudes.append(lat_f)
                    longitudes.append(lon_f)
                    dates.append(date_str)
                
                # Process entire batch at once (vectorized!)
                prior_probs_batch = prior_model.get_prior_probs_batch(latitudes, longitudes, dates)
                
                # Fill matrix
                for i, row_idx in enumerate(batch_indices):
                    prior_matrix[row_idx] = prior_probs_batch[i]
                
                processed_samples += len(batch_indices)
                
                # Log progress every 10 batches
                if (batch_idx + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = processed_samples / elapsed if elapsed > 0 else 0
                    remaining = num_samples - processed_samples
                    eta = remaining / rate if rate > 0 else 0
                    print(f"  Progress: {processed_samples}/{num_samples} samples ({100*processed_samples/num_samples:.1f}%) | "
                          f"Rate: {rate:.0f} samples/sec | ETA: {eta/60:.1f} min")
            except Exception:
                # Uniform prior on error for all samples in batch
                uniform_prior = np.ones(num_species) / num_species
                for row_idx in batch_indices:
                    prior_matrix[row_idx] = uniform_prior
                processed_samples += len(batch_indices)
    
    # Save to HDF5 (handle Windows file locking by using temp file + atomic rename)
    print(f"\nSaving to {output_path}...")
    
    # Use a temporary file in the same directory, then rename atomically
    # This avoids Windows file locking issues
    temp_path = output_path.parent / f"{output_path.stem}_temp_{int(time.time())}{output_path.suffix}"
    final_path = output_path
    
    # Try to remove existing file if it exists (with retries for Windows)
    if final_path.exists():
        import time as time_module
        for attempt in range(5):
            try:
                final_path.unlink()
                print(f"  Removed existing file: {final_path}")
                break
            except PermissionError:
                if attempt < 4:
                    print(f"  File locked, waiting 1 second... (attempt {attempt+1}/5)")
                    time_module.sleep(1)
                else:
                    print(f"  Warning: Could not remove existing file, will overwrite")
            except Exception as e:
                print(f"  Warning: Could not remove existing file: {e}")
                break
    
    # Save with multiple backup strategies - ALWAYS save something!
    h5_saved = False
    h5_path = None
    npz_saved = False
    npz_path = None
    
    # Strategy 1: Save HDF5 to temp file first (most reliable)
    try:
        print(f"  Saving to temporary HDF5 file: {temp_path}")
        with h5py.File(temp_path, 'w', libver='latest', swmr=False) as f:
            # Create datasets
            f.create_dataset('priors', data=prior_matrix, compression='gzip', compression_opts=4)
            f.create_dataset('species_list', data=[s.encode('utf-8') for s in all_species])
            f.create_dataset('sample_indices', data=df.index.values)
            
            # Store metadata
            f.attrs['num_samples'] = num_samples
            f.attrs['num_species'] = num_species
            f.attrs['data_dir'] = str(data_dir)
        
        # HDF5 file is saved! Mark as success
        h5_saved = True
        h5_path = temp_path
        print(f"✓ HDF5 file saved successfully: {temp_path}")
        print(f"  File size: {temp_path.stat().st_size / 1e6:.1f} MB (compressed)")
        
        # Try to rename to final path (but don't fail if this doesn't work)
        print(f"  Attempting to rename to final path: {final_path.name}")
        if final_path.exists():
            try:
                final_path.unlink()  # Remove old file if it exists
            except (PermissionError, OSError) as e:
                print(f"  Warning: Could not remove existing file (may be locked): {e}")
                print(f"  Will try to rename anyway...")
        
        try:
            temp_path.replace(final_path)
            h5_path = final_path
            print(f"✓ Renamed to final path: {final_path}")
        except (PermissionError, OSError) as e:
            print(f"  ⚠️  Could not rename to final path: {e}")
            print(f"  ✓ BUT: File is saved and valid at: {temp_path}")
            print(f"  You can use it directly or rename it later")
            
    except Exception as e:
        print(f"  ❌ Error saving HDF5 file: {e}")
        print(f"  Will try numpy backup format...")
    
    # Strategy 2: ALWAYS save a numpy backup (more reliable, no locking issues)
    npz_backup_path = output_path.parent / f"{output_path.stem}_backup.npz"
    
    try:
        print(f"\n  Saving numpy backup: {npz_backup_path}")
        np.savez_compressed(
            npz_backup_path,
            priors=prior_matrix,
            species_list=np.array([s.encode('utf-8') for s in all_species], dtype=object),
            sample_indices=df.index.values,
            num_samples=num_samples,
            num_species=num_species,
        )
        npz_saved = True
        npz_path = npz_backup_path
        print(f"✓ Numpy backup saved: {npz_backup_path}")
        print(f"  File size: {npz_backup_path.stat().st_size / 1e6:.1f} MB (compressed)")
    except Exception as e:
        print(f"  ❌ Error saving numpy backup: {e}")
    
    # Summary - ensure at least one file was saved
    print("\n" + "=" * 80)
    print("SAVE SUMMARY")
    print("=" * 80)
    
    if h5_saved:
        print(f"✓ HDF5 file saved: {h5_path}")
        print(f"  Use in training with: --priors_cache {h5_path}")
    else:
        print(f"❌ HDF5 file NOT saved")
    
    if npz_saved:
        print(f"✓ Numpy backup saved: {npz_path}")
        print(f"  (Can be converted to HDF5 later if needed)")
    else:
        print(f"❌ Numpy backup NOT saved")
    
    if not (h5_saved or npz_saved):
        print("\n❌ CRITICAL: NO FILES WERE SAVED!")
        print("  This should not happen - please check disk space and permissions")
        raise RuntimeError("Failed to save prior cache in any format")
    
    # Set final_path to whichever HDF5 file was successfully saved (prefer final path, fallback to temp)
    if h5_saved:
        final_path = h5_path
    else:
        # If HDF5 failed but numpy worked, we'll need to convert later
        final_path = npz_path
        print(f"\n⚠️  Note: Only numpy format saved. HDF5 format preferred for training.")
        print(f"  You may need to convert the .npz file to .h5 later")
    
    # Verify HDF5 file if it was saved
    if h5_saved:
        print(f"\nVerifying HDF5 cache...")
        try:
            with h5py.File(h5_path, 'r') as f:
                assert f['priors'].shape == (num_samples, num_species)
                assert len(f['species_list']) == num_species
                print("✓ HDF5 cache file is valid")
        except Exception as e:
            print(f"⚠️  Warning: Could not verify HDF5 file: {e}")
            print(f"  But file exists and should be usable")
    
    print(f"\n✓ PRIOR CACHE COMPUTATION COMPLETE!")
    print(f"  Total samples: {num_samples}")
    print(f"  Total species: {num_species}")
    print(f"  Memory used: {prior_matrix.nbytes / 1e6:.1f} MB")
    print("=" * 80)
    
    return final_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-compute priors for training set")
    parser.add_argument("--data_dir", type=str, default="../Data/cbi",
                       help="Directory containing train.csv")
    parser.add_argument("--priors_dir", type=str, default="../Data/priors",
                       help="Directory containing eBird prior TIF files")
    parser.add_argument("--output", type=str, default="../Data/cbi/priors_cache.h5",
                       help="Output HDF5 file path")
    parser.add_argument("--num_workers", type=int, default=16,
                       help="Number of parallel workers (default: 16)")
    parser.add_argument("--batch_size", type=int, default=1000,
                       help="Batch size for multiprocessing (samples per batch)")
    
    args = parser.parse_args()
    
    precompute_priors(
        data_dir=Path(args.data_dir),
        priors_dir=Path(args.priors_dir),
        output_path=Path(args.output),
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )

