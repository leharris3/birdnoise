"""
Quick stress test of the prior cache file to verify it's not uniform.
"""
import sys
from pathlib import Path
import numpy as np
import h5py

cache_path = Path("../Data/cbi/priors_cache_temp_1766348290.h5")

if not cache_path.exists():
    print(f"❌ Cache file not found: {cache_path}")
    sys.exit(1)

print("=" * 80)
print("QUICK STRESS TEST: PRIOR CACHE FILE")
print("=" * 80)
print(f"\nLoading cache from: {cache_path}")

# Load cache
with h5py.File(cache_path, 'r') as f:
    prior_matrix = f['priors'][:]  # Load into memory
    cached_species = [s.decode('utf-8') for s in f['species_list']]
    sample_indices = f['sample_indices'][:]

print(f"✓ Loaded: {len(sample_indices)} samples, {len(cached_species)} species")
print(f"  Shape: {prior_matrix.shape}")

# Test a few random samples
num_test_samples = min(10, len(prior_matrix))
test_indices = np.random.choice(len(prior_matrix), num_test_samples, replace=False)

print(f"\n" + "=" * 80)
print(f"TESTING {num_test_samples} RANDOM SAMPLES")
print("=" * 80)

max_entropy = np.log(len(cached_species))
uniform_prob = 1.0 / len(cached_species)

results = []

for i, idx in enumerate(test_indices, 1):
    prior_vec = prior_matrix[idx]
    
    # Statistics
    max_prob = prior_vec.max()
    min_prob = prior_vec[prior_vec > 1e-10].min() if (prior_vec > 1e-10).any() else 0
    non_zero_count = (prior_vec > 1e-10).sum()
    significant_count = (prior_vec > 0.01).sum()  # > 1% probability
    
    # Entropy
    entropy = -np.sum(prior_vec * np.log(prior_vec + 1e-10))
    info_ratio = 1 - (entropy / max_entropy)
    
    # Top 5 species
    top5_idx = np.argsort(prior_vec)[-5:][::-1]
    top5_probs = prior_vec[top5_idx]
    top5_species = [cached_species[j] for j in top5_idx]
    
    results.append({
        'idx': idx,
        'max_prob': max_prob,
        'entropy': entropy,
        'info_ratio': info_ratio,
        'non_zero': non_zero_count,
        'significant': significant_count,
    })
    
    print(f"\nSample {i} (index {idx}):")
    print(f"  Max probability: {max_prob:.4f} ({100*max_prob:.2f}%)")
    print(f"  Entropy: {entropy:.3f} / {max_entropy:.3f}")
    print(f"  Information ratio: {info_ratio:.3f} (0=uniform, 1=very informative)")
    print(f"  Non-zero species: {non_zero_count} / {len(cached_species)}")
    print(f"  Significant species (>1%): {significant_count}")
    print(f"  Top 5 species:")
    for rank, (species, prob) in enumerate(zip(top5_species, top5_probs), 1):
        print(f"    {rank}. {species:40s}: {prob:.4f} ({100*prob:.2f}%)")

# Summary statistics
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

max_probs = [r['max_prob'] for r in results]
entropies = [r['entropy'] for r in results]
info_ratios = [r['info_ratio'] for r in results]
non_zero_counts = [r['non_zero'] for r in results]

print(f"\nMax Probability:")
print(f"  Mean: {np.mean(max_probs):.4f} ({100*np.mean(max_probs):.2f}%)")
print(f"  Min: {np.min(max_probs):.4f} ({100*np.min(max_probs):.2f}%)")
print(f"  Max: {np.max(max_probs):.4f} ({100*np.max(max_probs):.2f}%)")
print(f"  Uniform would be: {uniform_prob:.4f} ({100*uniform_prob:.2f}%)")

print(f"\nInformation Ratio:")
print(f"  Mean: {np.mean(info_ratios):.3f}")
print(f"  Min: {np.min(info_ratios):.3f}")
print(f"  Max: {np.max(info_ratios):.3f}")
print(f"  Uniform would be: ~0.0")

print(f"\nNon-zero Species Count:")
print(f"  Mean: {np.mean(non_zero_counts):.1f}")
print(f"  Min: {np.min(non_zero_counts)}")
print(f"  Max: {np.max(non_zero_counts)}")
print(f"  Uniform would be: {len(cached_species)} (all species)")

# Final verdict
mean_info_ratio = np.mean(info_ratios)
mean_max_prob = np.mean(max_probs)

print("\n" + "=" * 80)
print("VERDICT")
print("=" * 80)

if mean_info_ratio > 0.3 and mean_max_prob > 0.05:
    print("✓✓✓ PRIORS ARE HIGHLY INFORMATIVE!")
    print("   The cache file is working correctly.")
elif mean_info_ratio > 0.15 and mean_max_prob > 0.02:
    print("✓✓ PRIORS ARE MODERATELY INFORMATIVE")
    print("   The cache file is working, but could be more informative.")
elif mean_info_ratio > 0.05:
    print("✓ PRIORS ARE SLIGHTLY INFORMATIVE")
    print("   The cache file is working, but priors are quite uniform.")
else:
    print("⚠️  PRIORS APPEAR TO BE UNIFORM")
    print("   This suggests a problem with the normalization or prior computation.")

print(f"\n  Mean info ratio: {mean_info_ratio:.3f}")
print(f"  Mean max prob: {mean_max_prob:.4f} ({100*mean_max_prob:.2f}%)")
print("=" * 80)

