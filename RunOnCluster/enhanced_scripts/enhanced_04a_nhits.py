"""
enhanced_04a_nhits.py
---------------------
N-HITS ONLY - WITH DETAILED RESIDUAL TRACKING
"""

import os
import pandas as pd
import numpy as np
import time
import warnings
import torch
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MAE
from utils_detailed_metrics import save_detailed_predictions

warnings.filterwarnings('ignore')
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

DATA_DIR = "../processed_data"
OUTPUT_DIR = "../outputs"
MAX_EVALS = 20

os.makedirs(OUTPUT_DIR, exist_ok=True)

EXPERIMENTS = ["with_wsi_solar", "no_wsi_solar", "with_wsi_wind", "no_wsi_wind"]
ACCELERATOR = "gpu" if torch.cuda.is_available() else "cpu"

def load_and_adapt_dl_data(experiment_name):
    try:
        X_train = pd.read_parquet(f"{DATA_DIR}/X_train_{experiment_name}.parquet")
        y_train = pd.read_parquet(f"{DATA_DIR}/y_train_{experiment_name}.parquet")
        X_test = pd.read_parquet(f"{DATA_DIR}/X_test_{experiment_name}.parquet")
        y_test = pd.read_parquet(f"{DATA_DIR}/y_test_{experiment_name}.parquet")
        
        start_date = "2022-01-01 00:00:00"
        train_dates = pd.date_range(start=start_date, periods=len(y_train), freq='H')
        test_dates = pd.date_range(start=train_dates[-1], periods=len(y_test) + 1, freq='H')[1:]
        
        X_train_exog = X_train.select_dtypes(include=[np.number])
        X_test_exog = X_test.select_dtypes(include=[np.number])

        df_train = X_train_exog.copy()
        df_train['ds'] = train_dates.values
        df_train['y'] = y_train.values.ravel()
        df_train['unique_id'] = 'ts_01'
        
        df_test = X_test_exog.copy()
        df_test['ds'] = test_dates.values
        df_test['y'] = y_test.values.ravel()
        df_test['unique_id'] = 'ts_01'
        
        exog_cols = [c for c in df_train.columns if c not in ['unique_id', 'ds', 'y']]
        
        return {'df_train': df_train, 'df_test': df_test, 'exog_cols': exog_cols, 'h': len(df_test)}
    except FileNotFoundError:
        print(f"[ERROR] Data not found")
        return None

def get_search_space():
    return {
        'learning_rate': hp.loguniform('learning_rate', np.log(1e-4), np.log(1e-2)),
        'max_steps': hp.choice('max_steps', [500, 1000]),
        'batch_size': hp.choice('batch_size', [32, 64, 128]),
        'n_blocks_choice': hp.choice('n_blocks_choice', [0, 1]),
        'mlp_units_choice': hp.choice('mlp_units_choice', [0, 1]),
        'dropout': hp.uniform('dropout', 0.0, 0.5)
    }

def train_and_eval(params, data):
    df_train = data['df_train']
    df_test = data['df_test']
    h = data['h']
    exog_cols = data['exog_cols']
    input_size = min(h, 96)
    
    try:
        n_blocks = [[1, 1, 1], [2, 2, 2]][params['n_blocks_choice']]
        mlp_units = [[[512, 512], [512, 512], [512, 512]], [[64, 64], [64, 64], [64, 64]]][params['mlp_units_choice']]
        
        model = NHITS(
            h=h,
            input_size=input_size,
            loss=MAE(),
            scaler_type='standard',
            max_steps=params['max_steps'],
            learning_rate=params['learning_rate'],
            batch_size=params['batch_size'],
            accelerator=ACCELERATOR,
            futr_exog_list=exog_cols,
            n_blocks=n_blocks,
            mlp_units=mlp_units,
            dropout_prob_theta=params['dropout']
        )

        nf = NeuralForecast(models=[model], freq='H')
        nf.fit(df=df_train, val_size=0)
        
        futr_df = df_test.drop(columns=['y'])
        forecasts = nf.predict(futr_df=futr_df)
        
        model_col = None
        for col in forecasts.columns:
            if col.upper() == 'NHITS':
                model_col = col
                break
        
        y_true = df_test['y'].values
        y_pred = forecasts[model_col].values
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return {'loss': rmse, 'status': STATUS_OK, 'model_obj': nf}

    except Exception as e:
        print(f"   [WARNING] Trial failed: {e}")
        return {'loss': float('inf'), 'status': STATUS_OK, 'model_obj': None}

def append_results(experiment, best_params, metrics):
    filename = f"{OUTPUT_DIR}/results_nhits.txt"
    header = ""
    if not os.path.exists(filename):
        header = "=" * 50 + "\nMODEL: NHITS\n" + "=" * 50 + "\n\n"
    
    report = f"""--- EXPERIMENT: {experiment.upper()} ---
Date: {time.strftime("%Y-%m-%d %H:%M:%S")}
Best Parameters: {best_params}
MAE : {metrics['mae']:.4f} MW
RMSE: {metrics['rmse']:.4f} MW
R2  : {metrics['r2']:.4f}
{'-' * 50}
"""
    with open(filename, "a") as f:
        f.write(header + report)
    print(f"   [REPORT] Saved to {filename}")

if __name__ == "__main__":
    print("=" * 80)
    print(f"   N-HITS ONLY (WITH DETAILED TRACKING) on {ACCELERATOR.upper()}")
    print("=" * 80)

    for experiment in EXPERIMENTS:
        print(f"\n>>> EXPERIMENT: {experiment.upper()} <<<")
        
        data = load_and_adapt_dl_data(experiment)
        if data is None: continue
        
        def objective(params):
            return train_and_eval(params, data)
        
        print(f"   [Tuning] Running {MAX_EVALS} trials...")
        space = get_search_space()
        trials = Trials()
        
        best_indices = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=MAX_EVALS, trials=trials)
        best_params = space_eval(space, best_indices)
        
        print("   [Training] Final Run...")
        final_run = train_and_eval(best_params, data)
        nf_model = final_run.get('model_obj', None)
        
        if nf_model is None:
            print(f"   [ERROR] Training failed. Skipping...")
            continue
        
        futr_df = data['df_test'].drop(columns=['y'])
        forecasts = nf_model.predict(futr_df=futr_df)
        
        model_col = None
        for col in forecasts.columns:
            if col.upper() == 'NHITS':
                model_col = col
                break
                
        y_pred = forecasts[model_col].values
        y_true = data['df_test']['y'].values

        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred)
        }
        print(f"   [Result] RMSE: {metrics['rmse']:.2f} | MAE: {metrics['mae']:.2f}")
        
        append_results(experiment, best_params, metrics)
        
        save_detailed_predictions(
            model_name='nhits',
            experiment_name=experiment,
            y_true=y_true,
            y_pred=y_pred,
            best_params=best_params,
            additional_info={'mae': metrics['mae'], 'rmse': metrics['rmse'], 'r2': metrics['r2']}
        )
    
    print("\n" + "=" * 80)
    print("N-HITS COMPLETE")
    print("=" * 80)

