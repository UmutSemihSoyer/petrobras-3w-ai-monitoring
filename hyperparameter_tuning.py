import os
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV
from sklearn.utils.class_weight import compute_sample_weight
import lightgbm as lgb
import xgboost as xgb
import joblib

PROCESSED_DIR = "data_processed"
MODELS_DIR = "models_saved"

def tune_hyperparameters():
    print("=" * 60)
    print(" HYPERPARAMETER TUNING FOR 3W CLASSIFIERS ")
    print("=" * 60)
    
    feature_path = os.path.join(PROCESSED_DIR, "extracted_features.parquet")
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Feature matrix not found at {feature_path}")
        
    df = pd.read_parquet(feature_path)
    drop_cols = ['target_class', 'base_class', 'is_transient', 'file_name', 'source_folder']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = df[feature_cols].values
    y = df['base_class'].values
    groups = df['file_name'].values if 'file_name' in df.columns else df.index.values
    
    X = np.nan_to_num(X, nan=0.0, posinf=1e8, neginf=-1e8)
    X = np.clip(X, -1e10, 1e10)
    
    sgkf = StratifiedGroupKFold(n_splits=3)
    
    # Define Parameter Grid for LightGBM
    lgb_param_grid = {
        'n_estimators': [100, 150, 200],
        'learning_rate': [0.03, 0.08, 0.1],
        'num_leaves': [31, 63, 127],
        'max_depth': [-1, 6, 10],
        'subsample': [0.8, 1.0]
    }
    
    print("\nRunning RandomizedSearchCV for LightGBM Classifier...")
    lgb_model = lgb.LGBMClassifier(random_state=42, class_weight='balanced', verbose=-1, n_jobs=-1)
    
    search = RandomizedSearchCV(
        estimator=lgb_model,
        param_distributions=lgb_param_grid,
        n_iter=5,
        scoring='f1_weighted',
        cv=sgkf,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    t0 = time.time()
    search.fit(X, y, groups=groups)
    elapsed = time.time() - t0
    
    print(f"\nHyperparameter tuning completed in {elapsed:.2f} seconds!")
    print(f"Best Weighted F1 Score: {search.best_score_:.4f}")
    print("Best Parameters:")
    for param, val in search.best_params_.items():
        print(f" - {param}: {val}")

if __name__ == "__main__":
    tune_hyperparameters()
