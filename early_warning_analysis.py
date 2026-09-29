import os
import glob
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from feature_engineering import extract_features_from_df

DATASET_DIR = "dataset"
MODELS_DIR = "models_saved"
OUTPUT_DIR = "reports_eda"
MODEL_PATH = os.path.join(MODELS_DIR, "best_3w_model.joblib")

os.makedirs(OUTPUT_DIR, exist_ok=True)

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

def analyze_early_warning():
    print("=" * 60)
    print(" EARLY WARNING SYSTEM: TRANSIENT LEAD-TIME ANALYSIS ")
    print("=" * 60)
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        
    model_data = joblib.load(MODEL_PATH)
    model = model_data['model']
    feature_cols = model_data['feature_cols']
    
    print(f"Loaded AI Model: {model_data.get('model_name')} (Accuracy: {model_data.get('accuracy', 0)*100:.2f}%)")
    
    # Analyze transient classes (Classes 1, 2, 5, 6, 7, 8, 9)
    transient_classes = [1, 2, 5, 6, 7, 8, 9]
    
    lead_time_results = []
    
    for c in transient_classes:
        folder_path = os.path.join(DATASET_DIR, str(c))
        if not os.path.exists(folder_path):
            continue
            
        files = glob.glob(os.path.join(folder_path, "*.parquet"))
        print(f"\nAnalyzing Class {c} ({classes_info[c]}): {len(files)} files...")
        
        # Test on up to 15 files per transient class
        for f in files[:15]:
            try:
                df = pd.read_parquet(f)
                if 'class' not in df.columns or df['class'].dropna().empty:
                    continue
                    
                # Find ground truth fault start time (when class label becomes c or 100+c)
                # 100+c is transient start, c is steady-state fault
                transient_label = 100 + c
                
                # Ground truth indices
                transient_indices = df.index[df['class'] == transient_label]
                fault_indices = df.index[df['class'] == c]
                
                if len(fault_indices) == 0 and len(transient_indices) == 0:
                    continue
                    
                # Ground truth start timestamp
                if len(transient_indices) > 0:
                    t_ground_truth = transient_indices[0]
                else:
                    t_ground_truth = fault_indices[0]
                    
                # Extract sliding window features (60s windows, 10s steps)
                feat_df = extract_features_from_df(df, window_size=60, step_size=10)
                if feat_df.empty:
                    continue
                    
                X_feats = feat_df[feature_cols].values
                X_cleaned = np.nan_to_num(X_feats, nan=0.0, posinf=1e8, neginf=-1e8)
                X_cleaned = np.clip(X_cleaned, -1e10, 1e10)
                
                preds = model.predict(X_cleaned)
                
                # Find first index where AI predicted an anomaly (pred != 0)
                anomaly_windows = np.where(preds != 0)[0]
                
                if len(anomaly_windows) > 0:
                    first_anomaly_win = anomaly_windows[0]
                    # Estimate timestamp of first anomaly detection
                    # Since window_size=60, step_size=10
                    point_idx = min(first_anomaly_win * 10, len(df) - 1)
                    t_ai_detect = df.index[point_idx]
                    
                    # Calculate lead time in seconds / minutes
                    if hasattr(t_ground_truth, 'timestamp') and hasattr(t_ai_detect, 'timestamp'):
                        lead_time_sec = (t_ground_truth - t_ai_detect).total_seconds()
                    else:
                        # Fallback for integer index
                        lead_time_sec = float(df.index.get_loc(t_ground_truth) - point_idx)
                        
                    lead_time_min = lead_time_sec / 60.0
                    
                    lead_time_results.append({
                        "class_id": c,
                        "class_name": classes_info[c],
                        "file_name": os.path.basename(f),
                        "t_ground_truth": str(t_ground_truth),
                        "t_ai_detect": str(t_ai_detect),
                        "lead_time_sec": lead_time_sec,
                        "lead_time_min": lead_time_min,
                        "detected_early": lead_time_sec > 0
                    })
            except Exception as e:
                print(f"Error evaluating {f}: {e}")

    df_lead = pd.DataFrame(lead_time_results)
    
    if df_lead.empty:
        print("No lead-time samples evaluated.")
        return
        
    print("\n" + "=" * 60)
    print(" EARLY WARNING LEAD-TIME RESULTS ")
    print("=" * 60)
    
    # Filter positive lead times (detected before or right at onset)
    df_early = df_lead[df_lead['lead_time_min'] >= 0]
    
    summary = df_lead.groupby(['class_id', 'class_name']).agg(
        total_events=('file_name', 'count'),
        early_detections=('detected_early', 'sum'),
        avg_lead_min=('lead_time_min', lambda x: np.mean(np.maximum(0, x))),
        max_lead_min=('lead_time_min', lambda x: np.max(np.maximum(0, x)))
    ).reset_index()
    
    print(summary.to_string(index=False))
    
    avg_overall_lead = summary['avg_lead_min'].mean()
    print(f"\nOverall Average Early Warning Lead-Time: {avg_overall_lead:.2f} minutes!")
    
    # Save Lead-Time Visualization Plot
    plt.figure(figsize=(10, 6), dpi=150)
    sns.barplot(data=summary, x='avg_lead_min', y='class_name', palette='crest')
    plt.title("AI Early Warning Lead-Time per Event Class (Minutes in Advance)", fontsize=13, fontweight='bold')
    plt.xlabel("Average Lead Time (Minutes Before Failure)", fontweight='bold')
    plt.ylabel("Event Class", fontweight='bold')
    
    for idx, row in summary.iterrows():
        plt.text(row['avg_lead_min'] + 0.5, idx, f"{row['avg_lead_min']:.1f} min", va='center', fontweight='bold', color='#1E293B')
        
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "early_warning_lead_time.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"\nSaved Early Warning plot to: {plot_path}")

if __name__ == "__main__":
    analyze_early_warning()
