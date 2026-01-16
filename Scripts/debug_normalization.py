"""
Debug why high raw abundance (0.86) becomes low probability (3%) after normalization.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from train_weighted_fusion import EBirdPriorWrapper
from eBirdPrior import EBirdCOGPrior

# Test location: Austin, TX, week 18
test_lat, test_lon = 30.27, -97.74
test_week = 18

data_dir = Path("../Data/cbi")
priors_dir = Path("../Data/priors")
train_csv = data_dir / "train.csv"

df = pd.read_csv(train_csv)
all_species = sorted(df["species"].unique())
code_to_species = dict(zip(df['ebird_code'], df['species']))
species_to_code = dict(zip(df['species'], df['ebird_code']))

# Get raw abundances for all species
print("=" * 80)
print("DEBUGGING NORMALIZATION: AUSTIN, WEEK 18")
print("=" * 80)

raw_prior = EBirdCOGPrior(str(priors_dir), product="abundance", resolution="27km")

# Get raw values for all species
print("\nGetting raw abundance values for all species...")
raw_values = {}
valid_species = []

for species in all_species:
    ebird_code = species_to_code.get(species)
    if ebird_code:
        try:
            raw_val = raw_prior.prob(ebird_code, test_lat, test_lon, week_idx=test_week, method="area", radius_km=5.0)
            raw_values[species] = raw_val
            if raw_val > 1e-10:
                valid_species.append((species, raw_val))
        except:
            pass

# Sort by raw value
valid_species.sort(key=lambda x: x[1], reverse=True)

print(f"\nFound {len(valid_species)} species with non-zero abundance")
print(f"\nTop 20 species by raw abundance:")
for i, (species, raw_val) in enumerate(valid_species[:20], 1):
    code = species_to_code.get(species, "?")
    print(f"  {i:2d}. {species:40s} ({code:8s}): {raw_val:.6f}")

# Check amecro
amecro_species = code_to_species.get('amecro')
if amecro_species in raw_values:
    amecro_raw = raw_values[amecro_species]
    amecro_rank = next((i for i, (s, _) in enumerate(valid_species, 1) if s == amecro_species), None)
    print(f"\nAmerican Crow (amecro):")
    print(f"  Raw abundance: {amecro_raw:.6f}")
    print(f"  Rank: {amecro_rank} / {len(valid_species)}")

# Now check what the normalization does
print("\n" + "=" * 80)
print("NORMALIZATION ANALYSIS")
print("=" * 80)

# Get normalized values
prior_model = EBirdPriorWrapper(
    priors_dir=priors_dir,
    species_list=all_species,
    cbi_metadata_path=train_csv,
)

prior_probs = prior_model.get_prior_probs_batch(
    [test_lat], [test_lon], ["2024-05-15"], sample_indices=None
)[0]

# Find which species have significant abundance (>1% of max)
max_raw = max(raw_values.values()) if raw_values else 0
threshold = max(1e-6, 0.01 * max_raw)

significant_species = [(s, v) for s, v in valid_species if v > threshold]
print(f"\nSpecies with significant abundance (>1% of max = {threshold:.6f}): {len(significant_species)}")

# Check normalization
if amecro_species in raw_values:
    amecro_raw = raw_values[amecro_species]
    amecro_idx = all_species.index(amecro_species)
    amecro_norm = prior_probs[amecro_idx]
    
    # Sum of all significant species
    sum_significant = sum(v for _, v in valid_species if v > threshold)
    expected_norm = amecro_raw / sum_significant if sum_significant > 0 else 0
    
    print(f"\nAmerican Crow normalization:")
    print(f"  Raw abundance: {amecro_raw:.6f}")
    print(f"  Normalized probability: {amecro_norm:.6f}")
    print(f"  Expected (if normalizing across significant): {expected_norm:.6f}")
    print(f"  Sum of significant species: {sum_significant:.6f}")
    print(f"  Number of significant species: {len(significant_species)}")
    
    if abs(amecro_norm - expected_norm) < 0.001:
        print(f"  ✓ Normalization matches expected (normalizing across {len(significant_species)} species)")
    else:
        print(f"  ⚠️  Normalization doesn't match expected - checking threshold...")

# Show distribution of raw values
print(f"\n" + "=" * 80)
print("RAW VALUE DISTRIBUTION")
print("=" * 80)

raw_vals = np.array([v for _, v in valid_species])
print(f"  Min: {raw_vals.min():.6f}")
print(f"  Max: {raw_vals.max():.6f}")
print(f"  Mean: {raw_vals.mean():.6f}")
print(f"  Median: {np.median(raw_vals):.6f}")
print(f"  Std: {raw_vals.std():.6f}")

# Histogram
bins = np.linspace(0, max_raw, 21)
hist, edges = np.histogram(raw_vals, bins=bins)
print(f"\n  Histogram:")
for i in range(len(hist)):
    if hist[i] > 0:
        print(f"    {edges[i]:.3f} - {edges[i+1]:.3f}: {hist[i]:3d} species")

print("=" * 80)

