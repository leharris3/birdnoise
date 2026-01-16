"""
Test specific case: Austin, TX, week 18 - should have amecro (American Crow) as dominant species.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import h5py

sys.path.insert(0, str(Path(__file__).parent))

from train_weighted_fusion import EBirdPriorWrapper

# Test location: Austin, TX in late spring (week 18)
test_lat, test_lon = 30.27, -97.74
test_date = "2024-05-15"  # Week 18

# Load species list
data_dir = Path("../Data/cbi")
priors_dir = Path("../Data/priors")
train_csv = data_dir / "train.csv"
cache_path = Path("../Data/cbi/priors_cache_temp_1766348290.h5")

df = pd.read_csv(train_csv)
all_species = sorted(df["species"].unique())

# Load eBird code mapping
code_to_species = dict(zip(df['ebird_code'], df['species']))
species_to_code = dict(zip(df['species'], df['ebird_code']))

print("=" * 80)
print("TEST: AUSTIN, TX - WEEK 18 (LATE SPRING)")
print("=" * 80)
print(f"\nLocation: Austin, TX ({test_lat}, {test_lon})")
print(f"Date: {test_date} (week ~18)")
print(f"Expected dominant species: amecro (American Crow)")

# Test 1: Direct lookup using EBirdPriorWrapper (on-the-fly)
print("\n" + "=" * 80)
print("TEST 1: Direct lookup (on-the-fly computation)")
print("=" * 80)

prior_model = EBirdPriorWrapper(
    priors_dir=priors_dir,
    species_list=all_species,
    cbi_metadata_path=train_csv,
)

prior_probs_direct = prior_model.get_prior_probs_batch(
    [test_lat], [test_lon], [test_date], sample_indices=None
)[0]

# Find amecro
amecro_species = code_to_species.get('amecro', None)
if amecro_species and amecro_species in all_species:
    amecro_idx = all_species.index(amecro_species)
    amecro_prob_direct = prior_probs_direct[amecro_idx]
    amecro_rank_direct = np.sum(prior_probs_direct > amecro_prob_direct) + 1
else:
    amecro_prob_direct = 0.0
    amecro_rank_direct = None

# Top 10
top10_idx = np.argsort(prior_probs_direct)[-10:][::-1]
top10_probs = prior_probs_direct[top10_idx]
top10_species = [all_species[i] for i in top10_idx]

print(f"\nTop 10 species (direct lookup):")
for i, (species, prob) in enumerate(zip(top10_species, top10_probs), 1):
    marker = " ⭐" if species == amecro_species else ""
    code = species_to_code.get(species, "?")
    print(f"  {i:2d}. {species:40s} ({code:8s}): {prob:.6f} ({100*prob:.2f}%){marker}")

if amecro_species:
    print(f"\nAmerican Crow (amecro):")
    print(f"  Probability: {amecro_prob_direct:.6f} ({100*amecro_prob_direct:.2f}%)")
    print(f"  Rank: {amecro_rank_direct} / {len(all_species)}")
    if amecro_rank_direct <= 10:
        print(f"  ✓✓✓ IN TOP 10 - GOOD!")
    elif amecro_rank_direct <= 50:
        print(f"  ⚠️  In top 50 but not top 10 - might be okay")
    else:
        print(f"  ❌ NOT IN TOP 50 - PROBLEM!")

# Test 2: Check raw eBird prior value
print("\n" + "=" * 80)
print("TEST 2: Raw eBird prior value (before normalization)")
print("=" * 80)

from eBirdPrior import EBirdCOGPrior
raw_prior = EBirdCOGPrior(str(priors_dir), product="abundance", resolution="27km")

# Get raw abundance for amecro
raw_amecro = raw_prior.prob("amecro", test_lat, test_lon, week_idx=18, method="area", radius_km=5.0)
print(f"\nRaw abundance for amecro: {raw_amecro:.6f}")

# Get raw abundance for a few other common species for comparison
test_species = ["rewbla", "amerob", "ameavo", "amecro"]
raw_values = {}
for sp in test_species:
    try:
        raw_val = raw_prior.prob(sp, test_lat, test_lon, week_idx=18, method="area", radius_km=5.0)
        raw_values[sp] = raw_val
        print(f"  {sp:8s}: {raw_val:.6f}")
    except:
        print(f"  {sp:8s}: ERROR")

# Test 3: Check cached prior (if we can find a sample at this location)
print("\n" + "=" * 80)
print("TEST 3: Cached prior (if sample exists at this location)")
print("=" * 80)

if cache_path.exists():
    # Load cache
    with h5py.File(cache_path, 'r') as f:
        cached_species = [s.decode('utf-8') for s in f['species_list']]
        sample_indices = f['sample_indices'][:]
        prior_matrix = f['priors']
    
    # Try to find a sample near Austin
    df_with_coords = df.copy()
    df_with_coords['lat'] = pd.to_numeric(df_with_coords['latitude'], errors='coerce')
    df_with_coords['lon'] = pd.to_numeric(df_with_coords['longitude'], errors='coerce')
    
    # Find samples within ~1 degree of Austin
    austin_samples = df_with_coords[
        (df_with_coords['lat'].between(test_lat - 1, test_lat + 1)) &
        (df_with_coords['lon'].between(test_lon - 1, test_lon + 1))
    ]
    
    if len(austin_samples) > 0:
        print(f"\nFound {len(austin_samples)} samples near Austin")
        # Get first sample's prior from cache
        sample_idx = austin_samples.index[0]
        if sample_idx in sample_indices:
            cache_row = np.where(sample_indices == sample_idx)[0][0]
            cached_prior = prior_matrix[cache_row]
            
            # Find amecro in cached prior
            if amecro_species in cached_species:
                amecro_idx_cached = cached_species.index(amecro_species)
                amecro_prob_cached = cached_prior[amecro_idx_cached]
                amecro_rank_cached = np.sum(cached_prior > amecro_prob_cached) + 1
            else:
                amecro_prob_cached = 0.0
                amecro_rank_cached = None
            
            # Top 10 from cache
            top10_idx_cached = np.argsort(cached_prior)[-10:][::-1]
            top10_probs_cached = cached_prior[top10_idx_cached]
            top10_species_cached = [cached_species[i] for i in top10_idx_cached]
            
            print(f"\nTop 10 species (from cache, sample index {sample_idx}):")
            for i, (species, prob) in enumerate(zip(top10_species_cached, top10_probs_cached), 1):
                marker = " ⭐" if species == amecro_species else ""
                code = species_to_code.get(species, "?")
                print(f"  {i:2d}. {species:40s} ({code:8s}): {prob:.6f} ({100*prob:.2f}%){marker}")
            
            if amecro_species:
                print(f"\nAmerican Crow (amecro) in cache:")
                print(f"  Probability: {amecro_prob_cached:.6f} ({100*amecro_prob_cached:.2f}%)")
                print(f"  Rank: {amecro_rank_cached} / {len(cached_species)}")
                if amecro_rank_cached <= 10:
                    print(f"  ✓✓✓ IN TOP 10 - GOOD!")
                elif amecro_rank_cached <= 50:
                    print(f"  ⚠️  In top 50 but not top 10 - might be okay")
                else:
                    print(f"  ❌ NOT IN TOP 50 - PROBLEM!")
        else:
            print(f"  Sample index {sample_idx} not found in cache")
    else:
        print(f"\nNo samples found near Austin in the dataset")

# Final verdict
print("\n" + "=" * 80)
print("VERDICT")
print("=" * 80)

if amecro_species:
    if amecro_rank_direct and amecro_rank_direct <= 10:
        print("✓✓✓ NORMALIZATION IS WORKING CORRECTLY")
        print(f"   American Crow is ranked #{amecro_rank_direct} with {100*amecro_prob_direct:.2f}% probability")
        print(f"   This is much higher than uniform (0.38%), so the prior is informative!")
    elif amecro_rank_direct and amecro_rank_direct <= 50:
        print("⚠️  NORMALIZATION MIGHT BE WORKING")
        print(f"   American Crow is ranked #{amecro_rank_direct} with {100*amecro_prob_direct:.2f}% probability")
        print(f"   Not in top 10, but still above uniform. May need investigation.")
    else:
        print("❌ NORMALIZATION MIGHT BE BROKEN")
        print(f"   American Crow is ranked #{amecro_rank_direct} with {100*amecro_prob_direct:.2f}% probability")
        print(f"   This is suspicious - should be much higher for Austin in spring!")
        print(f"   Raw value was: {raw_amecro:.6f}")
else:
    print("⚠️  Could not find American Crow in species list")

print("=" * 80)

