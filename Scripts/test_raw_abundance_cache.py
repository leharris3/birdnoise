"""
Test that the cache contains raw abundance scores (not normalized probabilities).
Specifically test Austin, week 18 where amecro should have ~0.86 raw abundance.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import h5py

sys.path.insert(0, str(Path(__file__).parent))

from train_weighted_fusion import EBirdPriorWrapper
from eBirdPrior import EBirdCOGPrior

# Test location: Austin, TX, week 18
test_lat, test_lon = 30.27, -97.74
test_date = "2024-05-15"  # Week 18

data_dir = Path("../Data/cbi")
priors_dir = Path("../Data/priors")
train_csv = data_dir / "train.csv"

# Try to find cache file
cache_files = list(data_dir.glob("priors_cache*.h5"))
if not cache_files:
    print("❌ No cache file found!")
    sys.exit(1)

cache_path = cache_files[0]  # Use first found cache file
print(f"Using cache file: {cache_path}")

df = pd.read_csv(train_csv)
all_species = sorted(df["species"].unique())
code_to_species = dict(zip(df['ebird_code'], df['species']))
species_to_code = dict(zip(df['species'], df['ebird_code']))

print("=" * 80)
print("TEST: RAW ABUNDANCE SCORES IN CACHE")
print("=" * 80)
print(f"\nLocation: Austin, TX ({test_lat}, {test_lon})")
print(f"Date: {test_date} (week ~18)")
print(f"Expected: amecro (American Crow) should have ~0.86 raw abundance")

# Get raw abundance from eBird prior directly (ground truth)
print("\n" + "=" * 80)
print("GROUND TRUTH: Raw abundance from eBird prior")
print("=" * 80)
raw_prior = EBirdCOGPrior(str(priors_dir), product="abundance", resolution="27km")
amecro_raw = raw_prior.prob("amecro", test_lat, test_lon, week_idx=18, method="area", radius_km=5.0)
print(f"amecro raw abundance: {amecro_raw:.6f}")

# Test cached prior
print("\n" + "=" * 80)
print("TEST: Cached prior (should match raw abundance)")
print("=" * 80)

prior_model = EBirdPriorWrapper(
    priors_dir=priors_dir,
    species_list=all_species,
    cbi_metadata_path=train_csv,
    cache_path=cache_path,
    df_index=None,
)

# Find a sample near Austin in the dataset
df_with_coords = df.copy()
df_with_coords['lat'] = pd.to_numeric(df_with_coords['latitude'], errors='coerce')
df_with_coords['lon'] = pd.to_numeric(df_with_coords['longitude'], errors='coerce')

austin_samples = df_with_coords[
    (df_with_coords['lat'].between(test_lat - 0.5, test_lat + 0.5)) &
    (df_with_coords['lon'].between(test_lon - 0.5, test_lon + 0.5))
]

if len(austin_samples) > 0:
    print(f"\nFound {len(austin_samples)} samples near Austin")
    sample_idx = austin_samples.index[0]
    sample_row = austin_samples.iloc[0]
    
    print(f"Using sample index {sample_idx}")
    print(f"  Location: {sample_row.get('location', 'N/A')}")
    print(f"  Date: {sample_row.get('date', 'N/A')}")
    
    # Get prior from cache
    cached_prior = prior_model.get_prior_probs_batch(
        [sample_row['lat']], [sample_row['lon']], [sample_row.get('date', test_date)],
        sample_indices=[sample_idx]
    )[0]
    
    # Find amecro
    amecro_species = code_to_species.get('amecro')
    if amecro_species and amecro_species in all_species:
        amecro_idx = all_species.index(amecro_species)
        amecro_cached = cached_prior[amecro_idx]
        
        print(f"\nAmerican Crow (amecro) in cache:")
        print(f"  Cached value: {amecro_cached:.6f}")
        print(f"  Raw abundance: {amecro_raw:.6f}")
        print(f"  Difference: {abs(amecro_cached - amecro_raw):.6f}")
        
        if abs(amecro_cached - amecro_raw) < 0.01:
            print(f"  ✓✓✓ MATCHES! Cache contains raw abundance scores")
        elif amecro_cached < 0.1:
            print(f"  ❌ TOO LOW! Cache appears to have normalized probabilities")
            print(f"     (Normalized would be ~0.03, raw should be ~0.86)")
        else:
            print(f"  ⚠️  Close but not exact match - may be due to coordinate differences")
    
    # Check a few other species
    print(f"\nTop 10 species from cache (should be raw abundance, not normalized):")
    top10_idx = np.argsort(cached_prior)[-10:][::-1]
    top10_values = cached_prior[top10_idx]
    top10_species = [all_species[i] for i in top10_idx]
    
    for i, (species, val) in enumerate(zip(top10_species, top10_values), 1):
        code = species_to_code.get(species, "?")
        # Get raw value for comparison
        try:
            raw_val = raw_prior.prob(code, test_lat, test_lon, week_idx=18, method="area", radius_km=5.0)
            match = "✓" if abs(val - raw_val) < 0.01 else "?"
            print(f"  {i:2d}. {species:40s} ({code:8s}): {val:.6f} (raw: {raw_val:.6f}) {match}")
        except:
            print(f"  {i:2d}. {species:40s} ({code:8s}): {val:.6f}")
    
    # Check if values look like raw abundance (high values) or normalized (low values)
    max_val = cached_prior.max()
    mean_val = cached_prior[cached_prior > 1e-6].mean()
    
    print(f"\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    print(f"Max value in cache: {max_val:.6f}")
    print(f"Mean non-zero value: {mean_val:.6f}")
    
    if max_val > 0.5:
        print(f"✓✓✓ Cache contains RAW ABUNDANCE SCORES (high values)")
        print(f"   This is correct - values should be in 0-1 range, not normalized")
    elif max_val < 0.1:
        print(f"❌ Cache appears to contain NORMALIZED PROBABILITIES (low values)")
        print(f"   This is wrong - should be raw abundance scores")
    else:
        print(f"⚠️  Uncertain - values are in middle range")
    
else:
    print("\nNo samples found near Austin - testing with direct lookup")
    cached_prior = prior_model.get_prior_probs_batch(
        [test_lat], [test_lon], [test_date], sample_indices=None
    )[0]
    
    amecro_species = code_to_species.get('amecro')
    if amecro_species and amecro_species in all_species:
        amecro_idx = all_species.index(amecro_species)
        amecro_cached = cached_prior[amecro_idx]
        
        print(f"\nAmerican Crow (amecro):")
        print(f"  Cached value: {amecro_cached:.6f}")
        print(f"  Raw abundance: {amecro_raw:.6f}")
        
        if abs(amecro_cached - amecro_raw) < 0.01:
            print(f"  ✓✓✓ MATCHES! Cache contains raw abundance scores")
        else:
            print(f"  Difference: {abs(amecro_cached - amecro_raw):.6f}")

print("=" * 80)

