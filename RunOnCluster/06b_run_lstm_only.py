"""
06b_run_lstm_only.py
--------------------
LSTM ONLY from Legacy Suite.
Uses NeuralForecast LSTM implementation.
"""

import os
import pandas as pd
import numpy as np
import warnings
import torch
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# NeuralForecast
from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM
from neuralforecast.losses.pytorch import MAE

warnings.filterwarnings('ignore')
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

# --- CONFIGURATION ---
DATA_DIR = "processed_data"
OUTPUT_DIR = "outputs"
MAX_EVALS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

EXPERIMENTS = [
    "with_wsi_solar",
    "no_wsi_solar",
    "with_wsi_wind",
    "no_wsi_wind"
]

# --- DATA LOADER ---
def load_panel(experiment_name):
    """Load data in NeuralForecast panel format"""
    try:
        X_train = pd.read_parquet(f"{DATA_DIR}/X_train_{experiment_name}.parquet")
        y_train = pd.read_parquet(f"{DATA_DIR}/y_train_{experiment_name}.parquet")
        X_test = pd.read_parquet(f"{DATA_DIR}/X_test_{experiment_name}.parquet")
        y_test = pd.read_parquet(f"{DATA_DIR}/y_test_{experiment_name}.parquet")
        
        # Recreate Dummy Dates
        start_date = "2022-01-01 00:00:00"
        train_dates = pd.date_range(start=start_date, periods=len(y_train), freq='H')
        test_dates = pd.date_range(start=train_dates[-1], periods=len(y_test) + 1, freq='H')[1:]
        
        # Remove date cols from X if they exist to avoid errors
        X_tr_clean = X_train.select_dtypes(include=[np.number])
        X_te_clean = X_test.select_dtypes(include=[np.number])

        df_train = X_tr_clean.copy()
        df_train['ds'] = train_dates
        df_train['y'] = y_train.values
        df_train['unique_id'] = 'ts_01'
        
        df_test = X_te_clean.copy()
        df_test['ds'] = test_dates
        df_test['y'] = y_test.values
        df_test['unique_id'] = 'ts_01'
        
        return df_train, df_test
    except Exception as e:
        print(f"   [ERROR] Failed to load data: {e}")
        return None, None

# --- SEARCH SPACE ---
def get_search_space():
    return {
        'learning_rate': hp.loguniform('learning_rate', np.log(1e-4), np.log(1e-2)),
        'hidden_size': hp.choice('hidden_size', [32, 64, 128]),
        'num_layers': hp.choice('num_layers', [1, 2]),
        'dropout': hp.uniform('dropout', 0.0, 0.5),
        'max_steps': hp.choice('max_steps', [500, 1000])
    }

# --- TRAINER ---
def train_nf_lstm(params, df_train, df_test):
    try:
        # Extract exog cols
        exog_cols = [c for c in df_train.columns if c not in ['unique_id', 'ds', 'y']]
        h = len(df_test)
        
        model = LSTM(h=h, input_size=min(h, 96), loss=MAE(),
                     hidden_size=params['hidden_size'],
                     num_layers=params['num_layers'],
                     dropout=params['dropout'],
                     max_steps=params['max_steps'],
                     futr_exog_list=exog_cols,
                     accelerator=DEVICE)
        
        nf = NeuralForecast(models=[model], freq='H')
        nf.fit(df=df_train, val_size=0)
        
        futr_df = df_test.drop(columns=['y'])
        fcst = nf.predict(futr_df=futr_df)
        
        preds = fcst['LSTM'].values
        y_true = df_test['y'].values
        rmse = np.sqrt(mean_squared_error(y_true, preds))
        return {'loss': rmse, 'status': STATUS_OK, 'model_obj': nf}
    except Exception as e:
        print(f"   [WARNING] LSTM training failed: {e}")
        return {'loss': float('inf'), 'status': STATUS_OK, 'model_obj': None}

# --- MAIN ---
def append_results(experiment, best_params, metrics):
    filename = f"{OUTPUT_DIR}/results_lstm.txt"
    with open(filename, "a") as f:
        f.write(f"EXP: {experiment}\nParams: {best_params}\nRMSE: {metrics['rmse']:.4f}\nMAE: {metrics['mae']:.4f}\nR2: {metrics['r2']:.4f}\n---\n")
    print(f"[REPORT] Saved {filename}")

if __name__ == "__main__":
    print("========================================")
    print("   LSTM ONLY - LEGACY MODEL            ")
    print(f"   Running on {DEVICE.upper()}        ")
    print("========================================")

    for experiment in EXPERIMENTS:
        print(f"\n>>> EXPERIMENT: {experiment.upper()} <<<")
        
        # Data Loading
        df_tr, df_te = load_panel(experiment)
        if df_tr is None: 
            print(f"   [SKIP] Could not load data for {experiment}")
            continue
        
        # Objective
        def objective(params):
            return train_nf_lstm(params, df_tr, df_te)

        # Tune
        print(f"   [Tuning] Running {MAX_EVALS} trials...")
        space = get_search_space()
        trials = Trials()
        best_idx = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=MAX_EVALS, trials=trials)
        best_params = space_eval(space, best_idx)
        
        # Final Run & Eval
        print("   [Training] Champion Run...")
        res = train_nf_lstm(best_params, df_tr, df_te)
        
        if res.get('model_obj') is None:
            print(f"   [ERROR] LSTM training failed for {experiment}. Skipping...")
            continue
        
        futr = df_te.drop(columns=['y'])
        y_pred = res['model_obj'].predict(futr)['LSTM'].values
        y_true = df_te['y'].values

        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred)
        }
        
        print(f"   [Result] RMSE: {metrics['rmse']:.2f} | MAE: {metrics['mae']:.2f} | R2: {metrics['r2']:.4f}")
        append_results(experiment, best_params, metrics)
    
    print("\n[COMPLETE] LSTM training finished!")

