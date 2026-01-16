"""
Quick sanity check: Verify the cache file has good entropy (is informative).
"""
import sys
from pathlib import Path
import numpy as np
import h5py
from tqdm import tqdm

# Find cache file (prefer temp file from latest run)
cache_files = sorted(Path("../Data/cbi").glob("priors_cache*.h5"), key=lambda p: p.stat().st_mtime, reverse=True)
if not cache_files:
    print("❌ No cache file found!")
    sys.exit(1)

cache_path = cache_files[0]  # Use most recent
print(f"Testing cache file: {cache_path}")
print(f"  File size: {cache_path.stat().st_size / (1024*1024):.1f} MB")

print("=" * 80)
print("QUICK SANITY CHECK: PRIOR CACHE ENTROPY")
print("=" * 80)

# Load cache
with h5py.File(cache_path, 'r') as f:
    prior_matrix = f['priors'][:]  # Load into memory
    cached_species = [s.decode('utf-8') for s in f['species_list']]
    sample_indices = f['sample_indices'][:]

print(f"\n✓ Loaded cache: {len(sample_indices)} samples, {len(cached_species)} species")
print(f"  Shape: {prior_matrix.shape}")

# Compute entropy for all samples
print("\nComputing entropy metrics...")
entropies = []
info_ratios = []
max_probs = []
num_non_zero = []

num_samples = len(prior_matrix)
max_entropy = np.log(len(cached_species))

for i in tqdm(range(num_samples), desc="Processing"):
    prior_vec = prior_matrix[i]
    
    # Normalize for entropy calculation (raw abundance scores need normalization)
    prior_sum = prior_vec.sum()
    if prior_sum > 1e-10:
        normalized = prior_vec / prior_sum
    else:
        normalized = np.ones_like(prior_vec) / len(prior_vec)  # Uniform fallback
    
    max_prob = normalized.max()  # Max normalized probability
    non_zero_count = (prior_vec > 1e-10).sum()
    
    # Entropy (on normalized probabilities)
    entropy = -np.sum(normalized * np.log(normalized + 1e-10))
    info_ratio = 1 - (entropy / max_entropy)
    
    entropies.append(entropy)
    info_ratios.append(info_ratio)
    max_probs.append(max_prob)
    num_non_zero.append(non_zero_count)

entropies = np.array(entropies)
info_ratios = np.array(info_ratios)
max_probs = np.array(max_probs)
num_non_zero = np.array(num_non_zero)

# Statistics
print("\n" + "=" * 80)
print("ENTROPY STATISTICS")
print("=" * 80)
print(f"Mean entropy: {entropies.mean():.3f} / {max_entropy:.3f}")
print(f"Median entropy: {np.median(entropies):.3f}")
print(f"Min entropy: {entropies.min():.3f} (most informative)")
print(f"Max entropy: {entropies.max():.3f} (least informative)")

print("\n" + "=" * 80)
print("INFORMATION RATIO STATISTICS")
print("=" * 80)
print(f"Mean info ratio: {info_ratios.mean():.3f} (0=uniform, 1=very informative)")
print(f"Median info ratio: {np.median(info_ratios):.3f}")
print(f"Min info ratio: {info_ratios.min():.3f}")
print(f"Max info ratio: {info_ratios.max():.3f}")

# Categorize
highly_inf = (info_ratios > 0.5).sum()
moderately_inf = ((info_ratios > 0.2) & (info_ratios <= 0.5)).sum()
low_inf = (info_ratios <= 0.2).sum()

print(f"\nInformativeness breakdown:")
print(f"  Highly informative (ratio > 0.5): {highly_inf} ({100*highly_inf/num_samples:.1f}%)")
print(f"  Moderately informative (0.2 < ratio <= 0.5): {moderately_inf} ({100*moderately_inf/num_samples:.1f}%)")
print(f"  Low informative (ratio <= 0.2): {low_inf} ({100*low_inf/num_samples:.1f}%)")

print("\n" + "=" * 80)
print("MAX PROBABILITY STATISTICS")
print("=" * 80)
print(f"Mean max probability: {max_probs.mean():.4f} ({100*max_probs.mean():.2f}%)")
print(f"Median max probability: {np.median(max_probs):.4f} ({100*np.median(max_probs):.2f}%)")
print(f"Min max probability: {max_probs.min():.4f} ({100*max_probs.min():.2f}%)")
print(f"Max max probability: {max_probs.max():.4f} ({100*max_probs.max():.2f}%)")
print(f"Uniform would be: {1.0/len(cached_species):.4f} ({100/len(cached_species):.2f}%)")

print("\n" + "=" * 80)
print("VERDICT")
print("=" * 80)
mean_info_ratio = info_ratios.mean()
mean_max_prob = max_probs.mean()

if mean_info_ratio > 0.3 and mean_max_prob > 0.05:
    print("✓✓✓ PRIORS ARE HIGHLY INFORMATIVE!")
    print("   The cache file is working correctly with good entropy.")
elif mean_info_ratio > 0.15 and mean_max_prob > 0.02:
    print("✓✓ PRIORS ARE MODERATELY INFORMATIVE")
    print("   The cache file is working, entropy is reasonable.")
elif mean_info_ratio > 0.05:
    print("✓ PRIORS ARE SLIGHTLY INFORMATIVE")
    print("   The cache file is working, but entropy could be better.")
else:
    print("⚠️  PRIORS APPEAR TO BE UNIFORM")
    print("   This suggests a problem - entropy is too low.")

print(f"\n  Mean info ratio: {mean_info_ratio:.3f}")
print(f"  Mean max prob: {mean_max_prob:.4f} ({100*mean_max_prob:.2f}%)")
print(f"  (Uniform would be: info_ratio=0.0, max_prob={100/len(cached_species):.2f}%)")
print("=" * 80)

