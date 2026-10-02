import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

PROCESSED_DIR = "data_processed"
MODELS_DIR = "models_saved"
OUTPUT_DIR = "reports_eda"

os.makedirs(MODELS_DIR, exist_ok=True)
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

import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight

def train_and_evaluate():
    print("=" * 60)
    print(" STEP 2: MACHINE LEARNING MODEL TRAINING & EVALUATION ")
    print("=" * 60)
    
    feature_path = os.path.join(PROCESSED_DIR, "extracted_features.parquet")
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Feature matrix not found at {feature_path}. Run feature_engineering.py first.")
        
    df = pd.read_parquet(feature_path)
    print(f"Loaded feature dataset: {df.shape}")
    
    # Exclude non-feature columns
    drop_cols = ['target_class', 'base_class', 'is_transient', 'file_name', 'source_folder']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = df[feature_cols].values
    y = df['base_class'].values
    groups = df['file_name'].values if 'file_name' in df.columns else df.index.values
    
    # Clean infinity and large values
    X = np.nan_to_num(X, nan=0.0, posinf=1e8, neginf=-1e8)
    X = np.clip(X, -1e10, 1e10)
    
    print(f"Features count: {len(feature_cols)}, Target classes: {len(np.unique(y))}")
    
    # Stratified Group Split (80% Train, 20% Test grouped by file_name to prevent data leakage)
    sgkf = StratifiedGroupKFold(n_splits=5)
    train_idx, test_idx = next(sgkf.split(X, y, groups=groups))
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"Train samples: {len(X_train)} (Groups: {len(np.unique(groups[train_idx]))}), Test samples: {len(X_test)} (Groups: {len(np.unique(groups[test_idx]))})")
    
    # Compute sample weights for class imbalance
    sample_weights = compute_sample_weight('balanced', y_train)
    
    models = {
        "XGBoost": xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, eval_metric='mlogloss'),
        "LightGBM": lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1, class_weight='balanced', verbose=-1),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=150, random_state=42, class_weight='balanced')
    }
    
    best_model = None
    best_f1 = -1.0
    best_name = ""
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        t0 = time.time()
        if name == "XGBoost":
            model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train, y_train)
        train_time = time.time() - t0
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results[name] = {
            "accuracy": acc,
            "f1_score": f1,
            "train_time_sec": train_time,
            "y_pred": y_pred
        }
        
        print(f"[{name}] Accuracy: {acc*100:.2f}%, Weighted F1: {f1:.4f}, Train Time: {train_time:.2f}s")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_name = name
            
    print("\n" + "=" * 60)
    print(f" BEST MODEL SELECTED: {best_name} (Weighted F1: {best_f1:.4f})")
    print("=" * 60)
    
    # Detailed classification report for best model
    best_pred = results[best_name]['y_pred']
    target_names = [f"{c}: {classes_info[c]}" for c in sorted(np.unique(y))]
    
    report_str = classification_report(y_test, best_pred, target_names=target_names)
    print("\nDetailed Classification Report:")
    print(report_str)
    
    # Save best model
    model_save_path = os.path.join(MODELS_DIR, "best_3w_model.joblib")
    joblib.dump({
        "model": best_model,
        "feature_cols": feature_cols,
        "classes_info": classes_info,
        "model_name": best_name,
        "accuracy": results[best_name]['accuracy'],
        "f1_score": best_f1
    }, model_save_path)
    print(f"Saved model to: {model_save_path}")
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, best_pred)
    plt.figure(figsize=(10, 8), dpi=150)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=[f"C{c}" for c in sorted(np.unique(y))],
        yticklabels=[f"C{c}" for c in sorted(np.unique(y))]
    )
    plt.title(f"3W Event Classification - Confusion Matrix ({best_name})", fontsize=13, fontweight='bold')
    plt.xlabel("Predicted Label", fontweight='bold')
    plt.ylabel("True Label", fontweight='bold')
    plt.tight_layout()
    
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved confusion matrix plot: {cm_path}")
    
    # Feature Importance
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1][:15] # top 15 features
        
        plt.figure(figsize=(10, 6), dpi=150)
        plt.barh(range(15), importances[indices][::-1], color='#3498db')
        plt.yticks(range(15), [feature_cols[i] for i in indices][::-1])
        plt.title(f"Top 15 Feature Importances ({best_name})", fontsize=13, fontweight='bold')
        plt.xlabel("Relative Importance", fontweight='bold')
        plt.tight_layout()
        
        fi_path = os.path.join(OUTPUT_DIR, "feature_importances.png")
        plt.savefig(fi_path)
        plt.close()
        print(f"Saved feature importances plot: {fi_path}")

if __name__ == "__main__":
    train_and_evaluate()
