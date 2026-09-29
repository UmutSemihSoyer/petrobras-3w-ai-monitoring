import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

PROCESSED_DIR = "data_processed"
MODELS_DIR = "models_saved"
OUTPUT_DIR = "reports_eda"
MODEL_PATH = os.path.join(MODELS_DIR, "best_3w_model.joblib")

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

def run_shap_analysis():
    print("=" * 60)
    print(" EXPLAINABLE AI (XAI) - SHAP VALUE ANALYSIS ")
    print("=" * 60)
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        
    model_data = joblib.load(MODEL_PATH)
    model = model_data['model']
    feature_cols = model_data['feature_cols']
    
    feature_path = os.path.join(PROCESSED_DIR, "extracted_features.parquet")
    df = pd.read_parquet(feature_path)
    
    X = df[feature_cols].values
    y = df['base_class'].values
    
    X = np.nan_to_num(X, nan=0.0, posinf=1e8, neginf=-1e8)
    X = np.clip(X, -1e10, 1e10)
    
    # Subsample 500 instances for fast & robust SHAP calculation
    sample_indices = np.random.choice(len(X), size=min(500, len(X)), replace=False)
    X_sample = X[sample_indices]
    y_sample = y[sample_indices]
    
    print(f"Calculating SHAP TreeExplainer values on {len(X_sample)} sample windows...")
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    print(f"SHAP calculation completed. Explainer output shape: {np.array(shap_values).shape}")
    
    # Save SHAP Summary Bar Plot
    plt.figure(figsize=(10, 8), dpi=150)
    # Handle list of arrays (multiclass) vs 3D array
    if isinstance(shap_values, list):
        mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    elif len(shap_values.shape) == 3: # (samples, features, classes)
        mean_shap = np.abs(shap_values).mean(axis=(0, 2))
    else:
        mean_shap = np.abs(shap_values).mean(axis=0)
        
    top_indices = np.argsort(mean_shap)[::-1][:15]
    top_cols = [feature_cols[i] for i in top_indices]
    top_vals = mean_shap[top_indices]
    
    plt.barh(range(15), top_vals[::-1], color='#10B981')
    plt.yticks(range(15), top_cols[::-1])
    plt.title("SHAP Global Feature Impact Across All 3W Events", fontsize=13, fontweight='bold')
    plt.xlabel("Mean |SHAP Value| (Impact on Model Decision)", fontweight='bold')
    plt.tight_layout()
    
    shap_plot_path = os.path.join(OUTPUT_DIR, "shap_feature_impact.png")
    plt.savefig(shap_plot_path)
    plt.close()
    print(f"Saved SHAP impact plot to: {shap_plot_path}")
    
    # Print top 5 root-cause features
    print("\nTop 5 Most Critical Sensor Features Identified by SHAP:")
    for rank, idx in enumerate(top_indices[:5], 1):
        print(f" {rank}. {feature_cols[idx]} (Avg Impact: {mean_shap[idx]:.4f})")

if __name__ == "__main__":
    run_shap_analysis()
