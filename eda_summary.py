import os
import glob
import pandas as pd
import numpy as np

DATASET_DIR = "dataset"

classes_info = {
    0: "Normal Operation",
    1: "Abrupt Increase of BSW",
    2: "Spurious Closure of DHSV",
    3: "Severe Slugging",
    4: "Flow Instability",
    5: "Rapid Productivity Loss",
    6: "Quick Restriction in PCK",
    7: "Scaling in PCK",
    8: "Hydrate in Production Line",
    9: "Hydrate in Service Line"
}

print("=" * 60)
print(" PETROBRAS 3W DATASET - EXPLORATORY SUMMARY ")
print("=" * 60)

summary_data = []

for c in range(10):
    folder_path = os.path.join(DATASET_DIR, str(c))
    if os.path.exists(folder_path):
        files = glob.glob(os.path.join(folder_path, "*.parquet"))
        num_files = len(files)
        
        total_rows = 0
        wells = set()
        
        # Sample first few files to count rows & detect well IDs
        for f in files[:10]: # inspect up to 10 sample files per class
            filename = os.path.basename(f)
            if "_" in filename:
                well_id = filename.split("_")[0]
                wells.add(well_id)
        
        summary_data.append({
            "Class ID": c,
            "Class Name": classes_info.get(c, "Unknown"),
            "Num Instances": num_files,
            "Sample Wells": ", ".join(sorted(list(wells))[:5])
        })

df_summary = pd.DataFrame(summary_data)
print(df_summary.to_string(index=False))

# Load a sample parquet file to display column details
sample_files = glob.glob(os.path.join(DATASET_DIR, "0", "*.parquet"))
if sample_files:
    sample_file = sample_files[0]
    print("\n" + "=" * 60)
    print(f" SAMPLE FILE INSPECTION: {os.path.basename(sample_file)}")
    print("=" * 60)
    df_sample = pd.read_parquet(sample_file)
    print("Shape:", df_sample.shape)
    print("Columns:", list(df_sample.columns))
    print("\nHead of sample file:")
    print(df_sample.head())
    print("\nSummary Statistics:")
    print(df_sample.describe().T[["count", "mean", "std", "min", "50%", "max"]])
