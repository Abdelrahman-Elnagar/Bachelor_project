"""
01_data_forge.py
----------------
THE UNIVERSAL TRUTH.
1. Loads raw CSVs (WSI and No-WSI).
2. Performs Time-Series Split (80% Train, 20% Test).
3. Engineers features for the 4 Experiments (Solar vs Wind).
4. Scales data (Fit on Train, Transform Test).
5. Saves processed data to .parquet for fast loading by model scripts.
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
# UPDATE THIS PATH to where your CSVs actually live on the cluster
DATA_DIR = "Bachelor abroad/training_data" 
OUTPUT_DIR = "processed_data"

FILES = {
    "with_wsi": "model_training_data_with_wsi.csv",
    "no_wsi": "model_training_data_no_wsi.csv"
}

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_and_save(filename, tag):
    print(f"\n[FORGE] Processing {filename} ({tag})...")
    
    # 1. Load Data
    path = os.path.join(DATA_DIR, filename)
    
    # Simple check to help you find the file if path is wrong
    if not os.path.exists(path):
        if os.path.exists(filename):
            path = filename
        else:
            raise FileNotFoundError(f"Cannot find {path}. Check your directory!")

    df = pd.read_csv(path)
    
    # 2. Time-Series Split (No Shuffling!)
    # We strictly use the first 80% for Train and last 20% for Test
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].reset_index(drop=True)
    test_df = df.iloc[train_size:].reset_index(drop=True)
    
    print(f"   Train Rows: {len(train_df)} | Test Rows: {len(test_df)}")

    # --- DEFINING THE 4 EXPERIMENTS ---
    # We need to create specific datasets for Solar vs Wind targets
    targets = ["solar_generation_mw", "wind_generation_mw"]
    
    for target in targets:
        # Identify the OTHER target to drop (Strict Feature Isolation)
        other_target = [t for t in targets if t != target][0]
        
        # Name: e.g., "wsi_solar" or "nowsi_wind"
        experiment_name = f"{tag}_{target.split('_')[0]}" 
        print(f"   -> Engineering Experiment: {experiment_name}")
        
        # PREPARE TRAIN
        # Drop the target itself AND the other target (so Wind doesn't help predict Solar)
        X_train = train_df.drop(columns=[target, other_target]).copy()
        y_train = train_df[target].copy()
        
        # PREPARE TEST
        X_test = test_df.drop(columns=[target, other_target]).copy()
        y_test = test_df[target].copy()
        
        # 3. Scaling (Fit on TRAIN only to prevent leakage)
        # We scale inputs (X) but typically keep Target (y) in MW for interpretable MAE
        scaler = StandardScaler()
        
        # Identify numeric columns (exclude dates/strings if any)
        num_cols = X_train.select_dtypes(include=[np.number]).columns
        
        # Learn stats from Train
        scaler.fit(X_train[num_cols])
        
        # Transform both
        X_train[num_cols] = scaler.transform(X_train[num_cols])
        X_test[num_cols] = scaler.transform(X_test[num_cols])
        
        # 4. Save to Parquet (Fast & Efficient)
        # We save X and y separately.
        # This creates the "Source of Truth" files on disk.
        X_train.to_parquet(f"{OUTPUT_DIR}/X_train_{experiment_name}.parquet")
        y_train.to_frame().to_parquet(f"{OUTPUT_DIR}/y_train_{experiment_name}.parquet")
        
        X_test.to_parquet(f"{OUTPUT_DIR}/X_test_{experiment_name}.parquet")
        y_test.to_frame().to_parquet(f"{OUTPUT_DIR}/y_test_{experiment_name}.parquet")
        
    print(f"   [OK] {tag} completed.")

# --- EXECUTION ---
if __name__ == "__main__":
    print("========================================")
    print("   PHASE 1: THE UNIVERSAL DATA FORGE    ")
    print("========================================")
    
    for tag, fname in FILES.items():
        try:
            process_and_save(fname, tag)
        except Exception as e:
            print(f"ERROR processing {fname}: {e}")

    print("\n[FORGE] Data preparation complete.")
    print(f"[FORGE] All files saved to: {os.path.abspath(OUTPUT_DIR)}")