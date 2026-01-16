"""
Quick Austin prior check: Show top 3, middle 3, and bottom 3 species.
"""
import sys
from pathlib import Path
import numpy as np
import h5py
import time
from train_weighted_fusion import EBirdPriorWrapper

# Austin coordinates
AUSTIN_LAT = 30.27
AUSTIN_LON = -97.74
AUSTIN_DATE = "2024-05-15"  # Spring (week 18)

print("=" * 80)
print("AUSTIN PRIOR CHECK")
print("=" * 80)
print(f"Coordinates: ({AUSTIN_LAT}, {AUSTIN_LON})")
print(f"Date: {AUSTIN_DATE} (Spring)")
print()

# Find most recent cache file
cache_files = sorted(Path("../Data/cbi").glob("priors_cache*.h5"), key=lambda p: p.stat().st_mtime, reverse=True)
if not cache_files:
    print("❌ No cache file found!")
    sys.exit(1)

cache_path = cache_files[0]
print(f"Using cache: {cache_path.name}")
print()

# Load cache to get species list
print("Loading cache...")
start_load = time.perf_counter()
with h5py.File(cache_path, 'r') as f:
    cached_species = [s.decode('utf-8') if isinstance(s, bytes) else s for s in f['species_list']]
load_time = time.perf_counter() - start_load
print(f"✓ Loaded cache in {load_time*1000:.1f} ms")
print()

# Initialize prior wrapper (need metadata for species mapping, but will use direct lookup)
print("Initializing prior model...")
cbi_metadata = Path("../Data/cbi/train.csv")
prior_model = EBirdPriorWrapper(
    priors_dir=Path("../Data/priors"),
    species_list=cached_species,
    cbi_metadata_path=cbi_metadata,
    cache_path=None,  # Use direct lookup, not cache
)
init_time = time.perf_counter() - start_load
print(f"✓ Initialized in {init_time*1000:.1f} ms")
print()

# Get prior probabilities (direct lookup, not from cache)
print("Computing prior probabilities...")
start_query = time.perf_counter()
prior_probs = prior_model.get_prior_probs_batch(
    [AUSTIN_LAT], [AUSTIN_LON], [AUSTIN_DATE],
    sample_indices=None  # Direct lookup
)[0]  # Get first (and only) result
query_time = time.perf_counter() - start_query
print(f"✓ Computed in {query_time*1000:.1f} ms")
print()

# Get species list
species_list = prior_model.species_list

# Sort by probability
sorted_indices = np.argsort(prior_probs)[::-1]  # Descending
sorted_probs = prior_probs[sorted_indices]
sorted_species = [species_list[i] for i in sorted_indices]

# Get top 3, middle 3, bottom 3
n = len(sorted_species)
top3_idx = sorted_indices[:3]
middle_start = n // 2 - 1
middle3_idx = sorted_indices[middle_start:middle_start+3]
bottom3_idx = sorted_indices[-3:]

print("=" * 80)
print("TOP 3 SPECIES (Highest Prior Probability)")
print("=" * 80)
for rank, idx in enumerate(top3_idx, 1):
    species = species_list[idx]
    prob = prior_probs[idx]
    print(f"  {rank}. {species:35s}: {prob:.6f} ({prob*100:.2f}%)")

print()
print("=" * 80)
print("MIDDLE 3 SPECIES (Median Prior Probability)")
print("=" * 80)
for rank, idx in enumerate(middle3_idx, 1):
    species = species_list[idx]
    prob = prior_probs[idx]
    print(f"  {rank}. {species:35s}: {prob:.6f} ({prob*100:.2f}%)")

print()
print("=" * 80)
print("BOTTOM 3 SPECIES (Lowest Prior Probability)")
print("=" * 80)
for rank, idx in enumerate(bottom3_idx, 1):
    species = species_list[idx]
    prob = prior_probs[idx]
    print(f"  {rank}. {species:35s}: {prob:.6f} ({prob*100:.2f}%)")

print()
print("=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)
print(f"Total species: {n}")
print(f"Max probability: {prior_probs.max():.6f} ({prior_probs.max()*100:.2f}%)")
print(f"Min probability: {prior_probs.min():.6f} ({prior_probs.min()*100:.2f}%)")
print(f"Mean probability: {prior_probs.mean():.6f} ({prior_probs.mean()*100:.2f}%)")
print(f"Non-zero species: {(prior_probs > 1e-8).sum()}")

# Entropy
normalized = prior_probs / prior_probs.sum()
entropy = -np.sum(normalized * np.log(normalized + 1e-10))
max_entropy = np.log(n)
info_ratio = 1 - (entropy / max_entropy)
print(f"Entropy: {entropy:.3f} / {max_entropy:.3f}")
print(f"Information ratio: {info_ratio:.3f} (0=uniform, 1=very informative)")

print()
print("=" * 80)
print("TIMING")
print("=" * 80)
print(f"Load time: {load_time*1000:.1f} ms")
print(f"Query time: {query_time*1000:.1f} ms")
print(f"Total time: {(load_time + query_time)*1000:.1f} ms")
print("=" * 80)

