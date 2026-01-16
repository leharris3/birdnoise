"""Check species mismatch between CBI dataset and prior database."""
from pathlib import Path
import pandas as pd

# Get species in prior database
priors_dir = Path("../Data/priors")
prior_codes = sorted({f.name.split("_")[0] for f in priors_dir.glob("*_uint8_cog.tif")})
print(f"Prior database: {len(prior_codes)} species")

# Get species in CBI
cbi_df = pd.read_csv("../Data/cbi/train.csv")
cbi_codes = sorted(cbi_df["ebird_code"].unique())
print(f"CBI dataset: {len(cbi_codes)} species")

# Find differences
missing_in_prior = set(cbi_codes) - set(prior_codes)
missing_in_cbi = set(prior_codes) - set(cbi_codes)
overlap = set(prior_codes) & set(cbi_codes)

print(f"\nOverlap: {len(overlap)} species")
print(f"Missing in prior: {len(missing_in_prior)} species")
if missing_in_prior:
    print(f"  Missing codes: {sorted(missing_in_prior)}")
    # Get species names for missing codes
    missing_species = cbi_df[cbi_df["ebird_code"].isin(missing_in_prior)][["species", "ebird_code"]].drop_duplicates()
    print(f"  Missing species:")
    for _, row in missing_species.iterrows():
        print(f"    {row['ebird_code']}: {row['species']}")

print(f"\nMissing in CBI: {len(missing_in_cbi)} species")
if missing_in_cbi:
    print(f"  Codes: {sorted(missing_in_cbi)}")

# Check coverage
coverage = len(overlap) / len(cbi_codes) * 100
print(f"\nCoverage: {coverage:.1f}% ({len(overlap)}/{len(cbi_codes)})")

# Check if all CBI species have prior files
print("\n=== Checking prior file existence for all CBI species ===")
missing_files = []
for code in cbi_codes:
    tif_path = priors_dir / f"{code}_abundance_27km_uint8_cog.tif"
    if not tif_path.exists():
        missing_files.append(code)

if missing_files:
    print(f"Missing prior files for {len(missing_files)} species:")
    for code in sorted(missing_files):
        species_name = cbi_df[cbi_df["ebird_code"] == code]["species"].iloc[0]
        print(f"  {code}: {species_name}")
else:
    print("✅ All CBI species have prior files!")

