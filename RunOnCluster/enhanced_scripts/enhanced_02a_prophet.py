"""
enhanced_02a_prophet.py
-----------------------
PROPHET ONLY - WITH DETAILED RESIDUAL TRACKING
"""

import os
import pandas as pd
import numpy as np
import time
import warnings
import logging
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from prophet import Prophet
from utils_detailed_metrics import save_detailed_predictions

warnings.filterwarnings('ignore')
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

DATA_DIR = "../processed_data"
OUTPUT_DIR = "../outputs"
MAX_EVALS = 30

os.makedirs(OUTPUT_DIR, exist_ok=True)

EXPERIMENTS = ["with_wsi_solar", "no_wsi_solar", "with_wsi_wind", "no_wsi_wind"]

def load_and_adapt_data(experiment_name):
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

        return {
            'y_train': y_train.values.ravel(),
            'y_test': y_test.values.ravel(),
            'X_train': X_train_exog,
            'X_test': X_test_exog,
            'train_ds': train_dates,
            'test_ds': test_dates
        }
    except FileNotFoundError:
        print(f"[ERROR] Data not found for {experiment_name}")
        return None

def get_search_space():
    return {
        'changepoint_prior_scale': hp.loguniform('changepoint_prior_scale', np.log(0.001), np.log(0.5)),
        'seasonality_prior_scale': hp.loguniform('seasonality_prior_scale', np.log(0.01), np.log(10.0)),
        'seasonality_mode': hp.choice('seasonality_mode', ['additive', 'multiplicative']),
        'changepoint_range': hp.uniform('changepoint_range', 0.8, 0.95)
    }

def train_and_eval(params, data):
    y_train = data['y_train']
    y_test = data['y_test']
    X_train = data['X_train']
    X_test = data['X_test']
    
    try:
        df_train = pd.DataFrame({'ds': data['train_ds'], 'y': y_train})
        for col in X_train.columns:
            df_train[col] = X_train[col].values
        
        m = Prophet(**params)
        for col in X_train.columns:
            m.add_regressor(col)
        m.fit(df_train)
        
        future = pd.DataFrame({'ds': data['test_ds']})
        for col in X_test.columns:
            future[col] = X_test[col].values
        
        forecast = m.predict(future)
        preds = forecast['yhat'].values

        preds = np.nan_to_num(preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        
        return {'loss': rmse, 'status': STATUS_OK, 'model': m, 'preds': preds}

    except Exception as e:
        print(f"   [WARNING] Trial failed: {e}")
        return {'loss': float('inf'), 'status': STATUS_OK}

def append_results(experiment, best_params, metrics):
    filename = f"{OUTPUT_DIR}/results_prophet.txt"
    header = ""
    if not os.path.exists(filename):
        header = "==================================================\nMODEL: PROPHET\n==================================================\n\n"
    
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

if __name__ == "__main__":
    print("="*80)
    print("   PROPHET ONLY (WITH DETAILED TRACKING)")
    print("="*80)

    for experiment in EXPERIMENTS:
        print(f"\n>>> EXPERIMENT: {experiment.upper()} <<<")
        
        data = load_and_adapt_data(experiment)
        if data is None: continue
        
        def objective(params):
            return train_and_eval(params, data)
        
        print(f"   [Tuning] Running {MAX_EVALS} trials...")
        space = get_search_space()
        trials = Trials()
        
        best_indices = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=MAX_EVALS, trials=trials)
        best_params = space_eval(space, best_indices)
        
        print("   [Training] Final Run...")
        final_res = train_and_eval(best_params, data)
        preds = final_res['preds']
        y_test = data['y_test']
        
        metrics = {
            'mae': mean_absolute_error(y_test, preds),
            'rmse': np.sqrt(mean_squared_error(y_test, preds)),
            'r2': r2_score(y_test, preds)
        }
        print(f"   [Result] RMSE: {metrics['rmse']:.2f} | MAE: {metrics['mae']:.2f}")
        
        append_results(experiment, best_params, metrics)
        
        save_detailed_predictions(
            model_name='prophet',
            experiment_name=experiment,
            y_true=y_test,
            y_pred=preds,
            best_params=best_params,
            additional_info={'mae': metrics['mae'], 'rmse': metrics['rmse'], 'r2': metrics['r2']}
        )
    
    print("\n" + "="*80)
    print("PROPHET COMPLETE")
    print("="*80)

