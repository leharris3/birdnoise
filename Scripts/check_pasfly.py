"""Check for pasfly and related flycatcher species."""
from pathlib import Path
import pandas as pd

# Get all prior codes
priors_dir = Path("../Data/priors")
prior_codes = sorted({f.name.split("_")[0] for f in priors_dir.glob("*.tif")})

# Get CBI data
cbi_df = pd.read_csv("../Data/cbi/train.csv")

# Check pasfly
pasfly_data = cbi_df[cbi_df["ebird_code"] == "pasfly"]
if len(pasfly_data) > 0:
    print("CBI pasfly data:")
    print(pasfly_data[["species", "ebird_code", "sci_name"]].drop_duplicates())

# Check all flycatcher species
print("\nAll flycatcher species in CBI:")
flycatchers = cbi_df[cbi_df["species"].str.contains("Flycatcher", case=False, na=False)]
print(flycatchers[["species", "ebird_code"]].drop_duplicates().sort_values("ebird_code"))

# Check which flycatcher codes are in priors
flycatcher_codes = flycatchers["ebird_code"].unique()
print(f"\nFlycatcher codes in priors:")
for code in sorted(flycatcher_codes):
    in_prior = code in prior_codes
    species = flycatchers[flycatchers["ebird_code"] == code]["species"].iloc[0]
    print(f"  {code}: {species} {'✅' if in_prior else '❌'}")

