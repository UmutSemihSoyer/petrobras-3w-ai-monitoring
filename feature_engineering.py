import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import time

DATASET_DIR = "dataset"
PROCESSED_DIR = "data_processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Key sensor columns to extract features from
SENSOR_COLS = [
    'P-PDG', 'T-PDG', 
    'P-TPT', 'T-TPT', 
    'P-MON-CKP', 'P-JUS-CKP', 
    'T-MON-CKP', 'T-JUS-CKP',
    'P-ANULAR', 'QGL', 'P-JUS-CKGL'
]

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

def extract_features_from_df(df, window_size=60, step_size=30):
    """
    Extracts statistical & dynamic time-series features using sliding windows.
    """
    # Create calculated columns if possible
    if 'P-MON-CKP' in df.columns and 'P-JUS-CKP' in df.columns:
        df['DELTA-P-CKP'] = df['P-MON-CKP'] - df['P-JUS-CKP']
    else:
        df['DELTA-P-CKP'] = np.nan
        
    if 'T-MON-CKP' in df.columns and 'T-JUS-CKP' in df.columns:
        df['DELTA-T-CKP'] = df['T-MON-CKP'] - df['T-JUS-CKP']
    else:
        df['DELTA-T-CKP'] = np.nan

    all_cols = SENSOR_COLS + ['DELTA-P-CKP', 'DELTA-T-CKP']
    
    rows = []
    n_points = len(df)
    
    # If the file is small, extract at least 1 summary window
    if n_points <= window_size:
        windows = [(0, n_points)]
    else:
        # Sample sliding windows across time series
        # To keep training fast and memory efficient, take up to 20 windows per file
        indices = np.linspace(0, n_points - window_size, min(20, (n_points - window_size) // step_size + 1), dtype=int)
        windows = [(idx, idx + window_size) for idx in indices]

    for start_idx, end_idx in windows:
        sub_df = df.iloc[start_idx:end_idx]
        feat_dict = {}
        
        # Get target class label (most common label in window)
        if 'class' in sub_df.columns and not sub_df['class'].dropna().empty:
            mode_label = sub_df['class'].dropna().mode()
            target_class = int(mode_label.iloc[0]) if not mode_label.empty else 0
        else:
            target_class = 0
            
        feat_dict['target_class'] = target_class
        feat_dict['base_class'] = target_class % 100 if target_class >= 100 else target_class
        feat_dict['is_transient'] = 1 if target_class >= 100 else 0
        
        for col in all_cols:
            if col in sub_df.columns:
                series = sub_df[col].dropna()
                if len(series) > 2:
                    feat_dict[f'{col}_mean'] = series.mean()
                    feat_dict[f'{col}_std'] = series.std()
                    feat_dict[f'{col}_min'] = series.min()
                    feat_dict[f'{col}_max'] = series.max()
                    feat_dict[f'{col}_range'] = series.max() - series.min()
                    
                    # Derivative / rate of change
                    diff = series.diff().dropna()
                    feat_dict[f'{col}_slope'] = diff.mean() if len(diff) > 0 else 0.0
                    feat_dict[f'{col}_slope_std'] = diff.std() if len(diff) > 1 else 0.0
                    feat_dict[f'{col}_is_missing'] = 0
                else:
                    feat_dict[f'{col}_mean'] = 0.0
                    feat_dict[f'{col}_std'] = 0.0
                    feat_dict[f'{col}_min'] = 0.0
                    feat_dict[f'{col}_max'] = 0.0
                    feat_dict[f'{col}_range'] = 0.0
                    feat_dict[f'{col}_slope'] = 0.0
                    feat_dict[f'{col}_slope_std'] = 0.0
                    feat_dict[f'{col}_is_missing'] = 1
            else:
                feat_dict[f'{col}_mean'] = 0.0
                feat_dict[f'{col}_std'] = 0.0
                feat_dict[f'{col}_min'] = 0.0
                feat_dict[f'{col}_max'] = 0.0
                feat_dict[f'{col}_range'] = 0.0
                feat_dict[f'{col}_slope'] = 0.0
                feat_dict[f'{col}_slope_std'] = 0.0
                feat_dict[f'{col}_is_missing'] = 1
                
        rows.append(feat_dict)
        
    return pd.DataFrame(rows)

def process_all_dataset():
    print("=" * 60)
    print(" STEP 1: FEATURE ENGINEERING & DATASET PREPROCESSING ")
    print("=" * 60)
    
    start_time = time.time()
    all_feature_dfs = []
    
    total_processed_files = 0
    
    for c in range(10):
        folder_path = os.path.join(DATASET_DIR, str(c))
        if not os.path.exists(folder_path):
            continue
            
        files = glob.glob(os.path.join(folder_path, "*.parquet"))
        print(f"Processing Class {c} ({classes_info[c]}): {len(files)} files...")
        
        # Limit per class to balance dataset and ensure fast extraction
        # Sample up to 50 files for Class 0 (normal), and up to 40 for others
        max_files = 50 if c == 0 else 40
        selected_files = files[:max_files]
        
        for f in selected_files:
            try:
                df = pd.read_parquet(f)
                feat_df = extract_features_from_df(df, window_size=120, step_size=60)
                feat_df['file_name'] = os.path.basename(f)
                feat_df['source_folder'] = c
                all_feature_dfs.append(feat_df)
                total_processed_files += 1
            except Exception as e:
                print(f"Error processing {f}: {e}")
                
    final_df = pd.concat(all_feature_dfs, ignore_index=True)
    
    # Fill any remaining NaNs with 0
    final_df = final_df.fillna(0.0)
    
    output_path = os.path.join(PROCESSED_DIR, "extracted_features.parquet")
    csv_path = os.path.join(PROCESSED_DIR, "extracted_features_sample.csv")
    
    final_df.to_parquet(output_path, engine='pyarrow', compression='brotli')
    final_df.head(100).to_csv(csv_path, index=False)
    
    elapsed = time.time() - start_time
    print(f"\nFeature extraction completed in {elapsed:.2f} seconds.")
    print(f"Processed files: {total_processed_files}")
    print(f"Extracted feature dataset shape: {final_df.shape}")
    print(f"Saved feature matrix to: {output_path}")
    
    # Print label distribution in feature dataset
    print("\nFeature Matrix Class Breakdown (base_class):")
    print(final_df['base_class'].value_counts().sort_index())

if __name__ == "__main__":
    process_all_dataset()
