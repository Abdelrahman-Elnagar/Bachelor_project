"""
enhanced_06a_random_forest.py
------------------------------
RANDOM FOREST ONLY - WITH DETAILED RESIDUAL TRACKING
"""

import os
import pandas as pd
import numpy as np
import time
import warnings
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils_detailed_metrics import save_detailed_predictions

warnings.filterwarnings('ignore')

DATA_DIR = "../processed_data"
OUTPUT_DIR = "../outputs"
MAX_EVALS = 20

os.makedirs(OUTPUT_DIR, exist_ok=True)

EXPERIMENTS = ["with_wsi_solar", "no_wsi_solar", "with_wsi_wind", "no_wsi_wind"]

def load_data(experiment_name):
    try:
        X_train_df = pd.read_parquet(f"{DATA_DIR}/X_train_{experiment_name}.parquet")
        X_test_df = pd.read_parquet(f"{DATA_DIR}/X_test_{experiment_name}.parquet")
        
        X_train = X_train_df.select_dtypes(include=[np.number]).values
        X_test = X_test_df.select_dtypes(include=[np.number]).values
        
        y_train = pd.read_parquet(f"{DATA_DIR}/y_train_{experiment_name}.parquet").values.ravel()
        y_test = pd.read_parquet(f"{DATA_DIR}/y_test_{experiment_name}.parquet").values.ravel()
        
        return X_train, y_train, X_test, y_test
    except:
        print(f"[ERROR] Data not found")
        return None, None, None, None

def get_search_space():
    return {
        'n_estimators': hp.choice('n_estimators', [100, 200, 500]),
        'max_depth': hp.choice('max_depth', [10, 20, None]),
        'min_samples_split': hp.choice('min_samples_split', [2, 5, 10]),
        'min_samples_leaf': hp.choice('min_samples_leaf', [1, 2, 4])
    }

def train_and_eval(params, X_train, y_train, X_test, y_test):
    model = RandomForestRegressor(**params, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return {'loss': rmse, 'status': STATUS_OK, 'model': model}

def append_results(experiment, best_params, metrics):
    filename = f"{OUTPUT_DIR}/results_random_forest.txt"
    with open(filename, "a") as f:
        f.write(f"EXP: {experiment}\nParams: {best_params}\nRMSE: {metrics['rmse']:.4f}\nMAE: {metrics['mae']:.4f}\nR2: {metrics['r2']:.4f}\n---\n")
    print(f"[REPORT] Saved {filename}")

if __name__ == "__main__":
    print("=" * 80)
    print("   RANDOM FOREST ONLY (WITH DETAILED TRACKING)")
    print("=" * 80)

    for experiment in EXPERIMENTS:
        print(f"\n>>> EXPERIMENT: {experiment.upper()} <<<")
        
        X_train, y_train, X_test, y_test = load_data(experiment)
        if X_train is None: continue
        
        def objective(params):
            return train_and_eval(params, X_train, y_train, X_test, y_test)
        
        print(f"   [Tuning] Running {MAX_EVALS} trials...")
        space = get_search_space()
        trials = Trials()
        best_idx = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=MAX_EVALS, trials=trials)
        best_params = space_eval(space, best_idx)
        
        print("   [Training] Champion Run...")
        res = train_and_eval(best_params, X_train, y_train, X_test, y_test)
        y_pred = res['model'].predict(X_test)
        y_true = y_test

        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred)
        }
        print(f"   [Result] RMSE: {metrics['rmse']:.2f} | MAE: {metrics['mae']:.2f}")
        
        append_results(experiment, best_params, metrics)
        
        save_detailed_predictions(
            model_name='random_forest',
            experiment_name=experiment,
            y_true=y_true,
            y_pred=y_pred,
            best_params=best_params,
            additional_info={'mae': metrics['mae'], 'rmse': metrics['rmse'], 'r2': metrics['r2']}
        )
    
    print("\n" + "=" * 80)
    print("RANDOM FOREST COMPLETE")
    print("=" * 80)

