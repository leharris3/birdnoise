"""
Debug script to check raw abundance values from eBird prior before normalization.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from eBirdPrior import EBirdCOGPrior

# Test location: Austin, TX in late spring (week 18)
test_lat, test_lon = 30.27, -97.74
test_week = 18

# Test a few known species
test_species_codes = ["rewbla", "amerob", "ribgul", "amecro", "ameavo"]

priors_dir = Path("../Data/priors")
prior = EBirdCOGPrior(str(priors_dir), product="abundance", resolution="27km")

print("=" * 80)
print("RAW ABUNDANCE VALUES FROM EBIRD PRIOR (before normalization)")
print("=" * 80)
print(f"\nLocation: Austin, TX ({test_lat}, {test_lon})")
print(f"Week: {test_week} (late spring)")

raw_values = {}
for species_code in test_species_codes:
    try:
        # Get raw abundance value
        raw_val = prior.prob(species_code, test_lat, test_lon, week_idx=test_week, method="point")
        raw_values[species_code] = raw_val
        print(f"{species_code:10s}: {raw_val:.6f}")
    except Exception as e:
        print(f"{species_code:10s}: ERROR - {e}")

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

if len(raw_values) > 0:
    values = np.array(list(raw_values.values()))
    print(f"Min raw value: {values.min():.6f}")
    print(f"Max raw value: {values.max():.6f}")
    print(f"Mean raw value: {values.mean():.6f}")
    print(f"Std raw value: {values.std():.6f}")
    
    if values.std() < 0.001:
        print("\n⚠️  WARNING: All raw values are nearly identical!")
        print("   This suggests the eBird prior data might be uniform, or")
        print("   we need to use a different method (e.g., 'area' instead of 'point')")
    else:
        print("\n✓ Raw values show variation - normalization should preserve this")
        
    # Test with area method (radius average)
    print("\n" + "=" * 80)
    print("TESTING WITH 'area' METHOD (5km radius average)")
    print("=" * 80)
    area_values = {}
    for species_code in test_species_codes:
        try:
            area_val = prior.prob(species_code, test_lat, test_lon, week_idx=test_week, method="area", radius_km=5.0)
            area_values[species_code] = area_val
            print(f"{species_code:10s}: {area_val:.6f}")
        except Exception as e:
            print(f"{species_code:10s}: ERROR - {e}")
    
    if len(area_values) > 0:
        area_vals = np.array(list(area_values.values()))
        print(f"\nArea method - Min: {area_vals.min():.6f}, Max: {area_vals.max():.6f}, Std: {area_vals.std():.6f}")
        
        if area_vals.std() > values.std():
            print("✓ Area method shows MORE variation - consider using this!")

print("=" * 80)

