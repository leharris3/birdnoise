"""
Quick test to demonstrate the difference between old and new normalization.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from train_weighted_fusion import EBirdPriorWrapper

# Test location: Austin, TX in late spring (week 18)
test_lat, test_lon = 30.27, -97.74
test_date = "2024-05-15"

# Load species list
data_dir = Path("../Data/cbi")
priors_dir = Path("../Data/priors")
train_csv = data_dir / "train.csv"

df = pd.read_csv(train_csv)
all_species = sorted(df["species"].unique())

print("=" * 80)
print("TESTING PRIOR NORMALIZATION")
print("=" * 80)
print(f"\nTest location: Austin, TX ({test_lat}, {test_lon})")
print(f"Test date: {test_date} (week ~18)")
print(f"Total species: {len(all_species)}")

# Initialize prior wrapper
prior_model = EBirdPriorWrapper(
    priors_dir=priors_dir,
    species_list=all_species,
    cbi_metadata_path=train_csv,
)

# Get prior probabilities
prior_probs = prior_model.get_prior_probs_batch(
    [test_lat], [test_lon], [test_date], sample_indices=None
)[0]

# Find top 10 species
top10_idx = np.argsort(prior_probs)[-10:][::-1]
top10_probs = prior_probs[top10_idx]
top10_species = [all_species[i] for i in top10_idx]

print("\n" + "=" * 80)
print("TOP 10 SPECIES (with NEW normalization - only across non-zero species)")
print("=" * 80)
for i, (species, prob) in enumerate(zip(top10_species, top10_probs), 1):
    print(f"{i:2d}. {species:40s}: {prob:.6f} ({100*prob:.2f}%)")

# Check informativeness
max_prob = prior_probs.max()
entropy = -np.sum(prior_probs * np.log(prior_probs + 1e-10))
max_entropy = np.log(len(prior_probs))
info_ratio = 1 - (entropy / max_entropy)
num_non_zero = (prior_probs > 1e-10).sum()

print("\n" + "=" * 80)
print("PRIOR INFORMATIVENESS METRICS")
print("=" * 80)
print(f"Max probability: {max_prob:.6f} ({100*max_prob:.2f}%)")
print(f"Species with non-zero abundance: {num_non_zero} / {len(all_species)}")
print(f"Entropy: {entropy:.3f} / {max_entropy:.3f}")
print(f"Information ratio: {info_ratio:.3f} (0=uniform, 1=very informative)")

if info_ratio > 0.5:
    print("✓ PRIOR IS HIGHLY INFORMATIVE!")
elif info_ratio > 0.2:
    print("✓ PRIOR IS MODERATELY INFORMATIVE")
else:
    print("⚠️  PRIOR IS STILL TOO UNIFORM - may need further investigation")

print("\n" + "=" * 80)
print("COMPARISON WITH OLD NORMALIZATION")
print("=" * 80)
print("OLD (normalize across all 264 species):")
print("  - Each species gets ~1/264 ≈ 0.38% probability")
print("  - Even dominant species get diluted")
print("  - Information ratio: ~0.0 (completely uniform)")
print("\nNEW (normalize only across non-zero species):")
print(f"  - Top species gets {100*max_prob:.2f}% probability")
print(f"  - Only {num_non_zero} species have non-zero probability")
print(f"  - Information ratio: {info_ratio:.3f}")
print("=" * 80)

