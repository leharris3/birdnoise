"""
Debug: Check what happens when we query all species at once.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from train_weighted_fusion import EBirdPriorWrapper

# Test location: Austin, TX in late spring
test_lat, test_lon = 30.27, -97.74
test_date = "2024-05-15"

data_dir = Path("../Data/cbi")
priors_dir = Path("../Data/priors")
train_csv = data_dir / "train.csv"

df = pd.read_csv(train_csv)
all_species = sorted(df["species"].unique())

prior_model = EBirdPriorWrapper(
    priors_dir=priors_dir,
    species_list=all_species,
    cbi_metadata_path=train_csv,
)

# Get prior probabilities
prior_probs = prior_model.get_prior_probs_batch(
    [test_lat], [test_lon], [test_date], sample_indices=None
)[0]

print("=" * 80)
print("DISTRIBUTION OF PRIOR PROBABILITIES (all 264 species)")
print("=" * 80)
print(f"\nLocation: Austin, TX ({test_lat}, {test_lon})")
print(f"Date: {test_date}")

# Statistics
non_zero = (prior_probs > 1e-10).sum()
zero = (prior_probs <= 1e-10).sum()
max_prob = prior_probs.max()
min_non_zero = prior_probs[prior_probs > 1e-10].min() if non_zero > 0 else 0

print(f"\nSpecies with non-zero probability: {non_zero} / {len(all_species)}")
print(f"Species with zero probability: {zero} / {len(all_species)}")
print(f"Max probability: {max_prob:.6f} ({100*max_prob:.2f}%)")
if non_zero > 0:
    print(f"Min non-zero probability: {min_non_zero:.6f} ({100*min_non_zero:.6f}%)")
    print(f"Ratio (max/min): {max_prob / min_non_zero:.2f}x")

# Histogram
print("\n" + "=" * 80)
print("HISTOGRAM OF PROBABILITIES")
print("=" * 80)
bins = np.linspace(0, max_prob, 21)
hist, edges = np.histogram(prior_probs, bins=bins)
for i in range(len(hist)):
    if hist[i] > 0:
        print(f"{edges[i]:.6f} - {edges[i+1]:.6f}: {hist[i]:4d} species")

# Top 20
print("\n" + "=" * 80)
print("TOP 20 SPECIES")
print("=" * 80)
top20_idx = np.argsort(prior_probs)[-20:][::-1]
top20_probs = prior_probs[top20_idx]
top20_species = [all_species[i] for i in top20_idx]

for i, (species, prob) in enumerate(zip(top20_species, top20_probs), 1):
    print(f"{i:2d}. {species:40s}: {prob:.6f} ({100*prob:.2f}%)")

# Check if rewbla/amecro are in there
code_to_species = dict(zip(df['ebird_code'], df['species']))
rewbla_species = code_to_species.get('rewbla', 'NOT FOUND')
amecro_species = code_to_species.get('amecro', 'NOT FOUND')

if rewbla_species in all_species:
    rewbla_idx = all_species.index(rewbla_species)
    rewbla_prob = prior_probs[rewbla_idx]
    rewbla_rank = np.sum(prior_probs > rewbla_prob) + 1
    print(f"\n{rewbla_species} (rewbla): prob={rewbla_prob:.6f}, rank={rewbla_rank}")

if amecro_species in all_species:
    amecro_idx = all_species.index(amecro_species)
    amecro_prob = prior_probs[amecro_idx]
    amecro_rank = np.sum(prior_probs > amecro_prob) + 1
    print(f"{amecro_species} (amecro): prob={amecro_prob:.6f}, rank={amecro_rank}")

print("=" * 80)

