"""
04_run_dl.py
----------------
THE DEEP LEARNING TITANS.
Models: N-HiTS, TFT, PatchTST.
Logic:
  1. Loads Parquet data.
  2. ADAPTS data: Converts to 'Panel' format (unique_id, ds, y) required by NeuralForecast.
  3. Uses Hyperopt to tune architecture (Layers, Heads, Patches).
  4. Saves results to outputs/.
"""

import os
import pandas as pd
import numpy as np
import time
import warnings
import torch
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# NeuralForecast Imports
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS, TFT, PatchTST
from neuralforecast.losses.pytorch import MAE, MSE

# Suppress warnings (PyTorch Lightning can be chatty)
warnings.filterwarnings('ignore')
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

# --- CONFIGURATION ---
DATA_DIR = "processed_data"
OUTPUT_DIR = "outputs"
MAX_EVALS = 20  # DL is slower; 20 trials is substantial. Increase if you have 48h+.
HORIZON_RATIO = 0.2 # We used 20% for testing in Forge. Matches that logic.

os.makedirs(OUTPUT_DIR, exist_ok=True)

EXPERIMENTS = [
    "with_wsi_solar",
    "no_wsi_solar",
    "with_wsi_wind",
    "no_wsi_wind"
]

MODELS = ["nhits", "tft", "patchtst"]

# Check GPU availability
ACCELERATOR = "gpu" if torch.cuda.is_available() else "cpu"
print(f"[SYSTEM] Running on {ACCELERATOR.upper()}")

# --- DATA ADAPTERS ---
def load_and_adapt_dl_data(experiment_name):
    """
    Loads parquet and reconstructs format for NeuralForecast.
    Requires: [unique_id, ds, y, ...exog_vars...]
    """
    try:
        X_train = pd.read_parquet(f"{DATA_DIR}/X_train_{experiment_name}.parquet")
        y_train = pd.read_parquet(f"{DATA_DIR}/y_train_{experiment_name}.parquet")
        X_test = pd.read_parquet(f"{DATA_DIR}/X_test_{experiment_name}.parquet")
        y_test = pd.read_parquet(f"{DATA_DIR}/y_test_{experiment_name}.parquet")
        
        # 1. Reconstruct Time Index
        start_date = "2022-01-01 00:00:00"
        
        # Try to find existing time col, else create synthetic
        time_col = None
        for col in X_train.columns:
            if 'time' in col.lower() or 'date' in col.lower():
                time_col = col
                break
        
        if time_col:
            train_dates = pd.to_datetime(X_train[time_col])
            test_dates = pd.to_datetime(X_test[time_col])
            # Exogenous vars shouldn't contain the time column or non-numeric types for DL
            X_train_exog = X_train.drop(columns=[time_col])
            X_test_exog = X_test.drop(columns=[time_col])
        else:
            # Synthetic Hourly Index
            train_dates = pd.date_range(start=start_date, periods=len(y_train), freq='H')
            test_dates = pd.date_range(start=train_dates[-1], periods=len(y_test) + 1, freq='H')[1:]
            X_train_exog = X_train
            X_test_exog = X_test
        
        # Safety: Keep ONLY numeric columns for exogenous variables
        X_train_exog = X_train_exog.select_dtypes(include=[np.number])
        X_test_exog = X_test_exog.select_dtypes(include=[np.number])

        # 2. Build Panel DataFrame (Long Format)
        # We assign a static unique_id since we have 1 series per experiment
        
        # Train Panel
        df_train = X_train_exog.copy()
        df_train['ds'] = train_dates.values
        df_train['y'] = y_train.values.ravel()
        df_train['unique_id'] = 'ts_01'
        
        # Test Panel
        df_test = X_test_exog.copy()
        df_test['ds'] = test_dates.values
        df_test['y'] = y_test.values.ravel()
        df_test['unique_id'] = 'ts_01'
        
        # Identify exogenous columns (all columns except required ones)
        exog_cols = [c for c in df_train.columns if c not in ['unique_id', 'ds', 'y']]
        
        return {
            'df_train': df_train,
            'df_test': df_test,
            'exog_cols': exog_cols,
            'h': len(df_test)  # Forecast Horizon matches test set length
        }

    except FileNotFoundError:
        print(f"[ERROR] Data not found for {experiment_name}")
        return None

def get_search_space(model_name):
    """Aggressive Search Spaces for Deep Learning Architectures"""
    
    common = {
        'learning_rate': hp.loguniform('learning_rate', np.log(1e-4), np.log(1e-2)),
        'max_steps': hp.choice('max_steps', [500, 1000]), # Iterations
        'batch_size': hp.choice('batch_size', [32, 64, 128])
    }
    
    if model_name == "nhits":
        return {
            **common,
            # Use integers instead of lists for simpler hyperopt handling
            'n_blocks_choice': hp.choice('n_blocks_choice', [0, 1]), # 0=[1,1,1], 1=[2,2,2]
            'mlp_units_choice': hp.choice('mlp_units_choice', [0, 1]), # 0=512, 1=64
            'dropout': hp.uniform('dropout', 0.0, 0.5)
        }
        
    elif model_name == "tft":
        return {
            **common,
            'hidden_size': hp.choice('hidden_size', [64, 128, 256]),
            'dropout': hp.uniform('dropout', 0.0, 0.5),
            'hidden_continuous_size': hp.choice('hidden_continuous_size', [16, 32, 64])
        }
        
    elif model_name == "patchtst":
        return {
            **common,
            'patch_len': hp.choice('patch_len', [16, 24]),
            'n_heads': hp.choice('n_heads', [4, 8]),
            'd_model': hp.choice('d_model', [64, 128]), # Latent dimension
            'dropout': hp.uniform('dropout', 0.0, 0.3)
        }
    return {}

def train_and_eval(params, model_name, data):
    """Objective Function"""
    df_train = data['df_train']
    df_test = data['df_test']
    h = data['h']
    exog_cols = data['exog_cols']
    
    # NeuralForecast usually forecasts a short horizon iteratively or directly.
    # If h is very large (e.g., 2000 hours), direct single-shot is hard.
    # We set input_size (lookback) relative to h or fixed window.
    input_size = min(h, 96) # Look back 96 hours (4 days) to predict
    
    try:
        # Initialize Model
        model = None
        
        # Base Args (common to all models)
        base_args = {
            'h': h,
            'input_size': input_size,
            'loss': MAE(),
            'scaler_type': 'standard',
            'max_steps': params['max_steps'],
            'learning_rate': params['learning_rate'],
            'batch_size': params['batch_size'],
            'accelerator': ACCELERATOR
        }
        
        if model_name == "nhits":
            # Convert choice indices to actual parameter lists
            # Use same structure for all: 3 stacks, varying blocks and units
            if params['n_blocks_choice'] == 0:
                n_blocks = [1, 1, 1]
            else:
                n_blocks = [2, 2, 2]
            
            if params['mlp_units_choice'] == 0:
                mlp_units = [[512, 512], [512, 512], [512, 512]]
            else:
                mlp_units = [[64, 64], [64, 64], [64, 64]]
            
            model = NHITS(
                **base_args,
                futr_exog_list=exog_cols,
                n_blocks=n_blocks,
                mlp_units=mlp_units,
                dropout_prob_theta=params['dropout']
            )
            
        elif model_name == "tft":
            model = TFT(
                **base_args,
                futr_exog_list=exog_cols,
                hidden_size=params['hidden_size'],
                dropout=params['dropout'],
                hidden_continuous_size=params['hidden_continuous_size']
            )
            
        elif model_name == "patchtst":
            # PatchTST does NOT support exogenous variables - remove them
            model = PatchTST(
                **base_args,
                # NO futr_exog_list for PatchTST!
                patch_len=params['patch_len'],
                n_heads=params['n_heads'],
                d_model=params['d_model'],
                dropout=params['dropout']
            )

        # Train
        nf = NeuralForecast(models=[model], freq='H')
        
        # We fit on Train without validation split (horizon is too large for practical val_size)
        # The models have max_steps limit to prevent overfitting
        nf.fit(df=df_train, val_size=0)
        
        # Predict
        # We need to pass the future exogenous variables found in df_test
        # For PatchTST (no exog support), futr_df will just have unique_id and ds
        if model_name == "patchtst":
            # PatchTST doesn't use exogenous variables
            futr_df = df_test[['unique_id', 'ds']].copy()
        else:
            futr_df = df_test.drop(columns=['y'])
        
        forecasts = nf.predict(futr_df=futr_df)
        
        # Extract predictions matching the test set
        # Model name in forecasts might be uppercase (NHITS, TFT, PatchTST)
        model_col = None
        for col in forecasts.columns:
            if col.upper() == model_name.upper():
                model_col = col
                break
        
        if model_col is None:
            raise ValueError(f"Cannot find prediction column for {model_name}. Available: {forecasts.columns.tolist()}")
        
        y_true = df_test['y'].values
        y_pred = forecasts[model_col].values
        
        # Evaluation
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        return {'loss': rmse, 'status': STATUS_OK, 'model_obj': nf}

    except Exception as e:
        print(f"   [WARNING] DL Trial failed: {e}")
        return {'loss': float('inf'), 'status': STATUS_OK, 'model_obj': None}

def append_results_to_file(model_name, experiment, best_params, metrics):
    filename = f"{OUTPUT_DIR}/results_{model_name}.txt"
    header = ""
    if not os.path.exists(filename):
        header = f"==================================================\nMODEL: {model_name.upper()}\n==================================================\n\n"
    
    report = f"""--- EXPERIMENT: {experiment.upper()} ---
Date: {time.strftime("%Y-%m-%d %H:%M:%S")}
Best Parameters: {best_params}
MAE : {metrics['mae']:.4f} MW
RMSE: {metrics['rmse']:.4f} MW
R2  : {metrics['r2']:.4f}
--------------------------------------------------
"""
    with open(filename, "a") as f:
        f.write(header + report)
    print(f"   [REPORT] Saved to {filename}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("========================================")
    print("   PHASE 3: THE DEEP LEARNING TITANS    ")
    print(f"   (N-HiTS / TFT / PatchTST) on {ACCELERATOR.upper()}")
    print("========================================")

    for model_name in MODELS:
        print(f"\n>>> ENTERING ARENA: {model_name.upper()} <<<")
        
        for experiment in EXPERIMENTS:
            print(f"\n   [Experiment] {experiment} ...")
            
            data = load_and_adapt_dl_data(experiment)
            if data is None: continue
            
            # Objective
            def objective(params):
                return train_and_eval(params, model_name, data)
            
            # Tuning
            print(f"   [Tuning] Running {MAX_EVALS} trials...")
            space = get_search_space(model_name)
            trials = Trials()
            
            best_indices = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=MAX_EVALS, trials=trials)
            best_params = space_eval(space, best_indices)
            
            # Final Run
            print("   [Training] Final Run with Champion Parameters...")
            final_run = train_and_eval(best_params, model_name, data)
            nf_model = final_run.get('model_obj', None)
            
            # Check if training succeeded
            if nf_model is None:
                print(f"   [ERROR] Final training failed for {experiment}. Skipping...")
                continue
            
            # Generate Final Predictions
            if model_name == "patchtst":
                # PatchTST doesn't use exogenous variables
                futr_df = data['df_test'][['unique_id', 'ds']].copy()
            else:
                futr_df = data['df_test'].drop(columns=['y'])
            
            forecasts = nf_model.predict(futr_df=futr_df)
            
            # Model name in forecasts might be uppercase (NHITS, TFT, PatchTST)
            model_col = None
            for col in forecasts.columns:
                if col.upper() == model_name.upper():
                    model_col = col
                    break
            
            if model_col is None:
                print(f"   [ERROR] Cannot find prediction column. Available: {forecasts.columns.tolist()}")
                continue
                
            y_pred = forecasts[model_col].values
            y_true = data['df_test']['y'].values
            
            metrics = {
                'mae': mean_absolute_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'r2': r2_score(y_true, y_pred)
            }
            print(f"   [Result] RMSE: {metrics['rmse']:.2f}")
            append_results_to_file(model_name, experiment, best_params, metrics)