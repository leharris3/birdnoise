"""Check BEANS-Zero CBI structure and matching with original CBI."""
from datasets import load_dataset
import numpy as np
import pandas as pd

# Load BEANS-Zero
print("Loading BEANS-Zero...")
ds = load_dataset("EarthSpeciesProject/BEANS-Zero", split="test")

# Filter to CBI
cbi_idx = np.where(np.array(ds["dataset_name"]) == "cbi")[0]
cbi_subset = ds.select(cbi_idx)
print(f"BEANS-Zero CBI test samples: {len(cbi_subset)}")

# Get unique labels
labels = sorted(set(cbi_subset["output"]))
print(f"Number of unique species: {len(labels)}")
print(f"Sample labels: {labels[:5]}")

# Check file_name format
print(f"\nFile name examples:")
for i in range(3):
    print(f"  {cbi_subset[i]['file_name']} -> {cbi_subset[i]['output']}")

# Load original CBI
print("\n--- Original CBI Dataset ---")
train_df = pd.read_csv("../Data/cbi/train.csv")
print(f"Original CBI train samples: {len(train_df)}")
print(f"Original unique species: {train_df['species'].nunique()}")

# Check if we can extract XC ID from BEANS file names
# BEANS file: XC168746.wav -> original: XC168746.mp3
beans_files = [s["file_name"].replace(".wav", "") for s in cbi_subset]
original_files = train_df["filename"].str.replace(".mp3", "").tolist()

# Find overlap
overlap = set(beans_files) & set(original_files)
print(f"\nFiles matching between BEANS test and original train: {len(overlap)}")

# Check if there's a test.csv
import os
test_csv_path = "../Data/cbi/test.csv"
if os.path.exists(test_csv_path):
    test_df = pd.read_csv(test_csv_path)
    print(f"\nOriginal CBI test.csv exists: {len(test_df)} rows")
    print(f"Columns: {list(test_df.columns)}")

# Check label mapping
print("\n--- Label Mapping Check ---")
beans_labels = set(labels)
original_labels = set(train_df["species"].unique())
print(f"Labels in both: {len(beans_labels & original_labels)}")
print(f"Only in BEANS: {beans_labels - original_labels}")
print(f"Only in original: {len(original_labels - beans_labels)} labels")

