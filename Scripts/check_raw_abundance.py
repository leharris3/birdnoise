"""
Check raw abundance values in cache (before normalization).
"""
import sys
from pathlib import Path
import numpy as np
import h5py

# Find most recent cache file
cache_files = sorted(Path("../Data/cbi").glob("priors_cache*.h5"), key=lambda p: p.stat().st_mtime, reverse=True)
cache_path = cache_files[0]
print(f"Checking raw abundance in: {cache_path}\n")

with h5py.File(cache_path, 'r') as f:
    prior_matrix = f['priors'][:]
    cached_species = [s.decode('utf-8') for s in f['species_list']]

print("=" * 80)
print("RAW ABUNDANCE STATISTICS (Before Normalization)")
print("=" * 80)

# Statistics on raw abundance
raw_max = prior_matrix.max(axis=1)  # Max raw abundance per sample
raw_mean = prior_matrix.mean(axis=1)  # Mean raw abundance per sample
raw_std = prior_matrix.std(axis=1)  # Std dev per sample

print(f"\nPer-sample statistics:")
print(f"  Mean max raw abundance: {raw_max.mean():.4f}")
print(f"  Median max raw abundance: {np.median(raw_max):.4f}")
print(f"  Min max raw abundance: {raw_max.min():.4f}")
print(f"  Max max raw abundance: {raw_max.max():.4f}")

print(f"\n  Mean of mean raw abundance: {raw_mean.mean():.4f}")
print(f"  Mean std dev: {raw_std.mean():.4f}")

# Check how many species have significant abundance per sample
significant_threshold = 0.1  # 10% of max abundance
num_significant = []
for i in range(len(prior_matrix)):
    max_val = prior_matrix[i].max()
    threshold = max_val * significant_threshold
    num_sig = (prior_matrix[i] > threshold).sum()
    num_significant.append(num_sig)

num_significant = np.array(num_significant)
print(f"\n  Mean # species with >10% of max abundance: {num_significant.mean():.1f}")
print(f"  Median # species: {np.median(num_significant):.1f}")

# Show example: sample with highest max abundance
best_idx = raw_max.argmax()
print(f"\n" + "=" * 80)
print(f"EXAMPLE: Sample with highest max abundance (index {best_idx})")
print("=" * 80)
sample = prior_matrix[best_idx]
top10_idx = np.argsort(sample)[-10:][::-1]
print(f"Top 10 species by raw abundance:")
for rank, idx in enumerate(top10_idx, 1):
    species = cached_species[idx]
    raw_val = sample[idx]
    print(f"  {rank:2d}. {species:30s}: {raw_val:.6f}")

# Show example: sample with moderate max abundance (median)
median_idx = np.argsort(raw_max)[len(raw_max)//2]
print(f"\n" + "=" * 80)
print(f"EXAMPLE: Sample with median max abundance (index {median_idx})")
print("=" * 80)
sample = prior_matrix[median_idx]
top10_idx = np.argsort(sample)[-10:][::-1]
print(f"Top 10 species by raw abundance:")
for rank, idx in enumerate(top10_idx, 1):
    species = cached_species[idx]
    raw_val = sample[idx]
    print(f"  {rank:2d}. {species:30s}: {raw_val:.6f}")

print("\n" + "=" * 80)
print("VERDICT")
print("=" * 80)
if raw_max.mean() > 0.5:
    print("✓✓✓ RAW ABUNDANCE SCORES ARE HIGHLY INFORMATIVE!")
    print("   Strong signal in raw values - good for fusion model.")
elif raw_max.mean() > 0.1:
    print("✓✓ RAW ABUNDANCE SCORES ARE MODERATELY INFORMATIVE")
    print("   Reasonable signal in raw values.")
else:
    print("⚠️  RAW ABUNDANCE SCORES ARE WEAK")
    print("   May need to check prior computation.")

print(f"\n  Mean max raw abundance: {raw_max.mean():.4f}")
print("=" * 80)

