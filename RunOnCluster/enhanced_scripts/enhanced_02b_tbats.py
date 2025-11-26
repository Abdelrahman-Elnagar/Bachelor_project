"""
enhanced_02b_tbats.py
---------------------
TBATS ONLY - WITH DETAILED RESIDUAL TRACKING
"""

import os
import pandas as pd
import numpy as np
import time
import warnings
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tbats import TBATS
from utils_detailed_metrics import save_detailed_predictions

warnings.filterwarnings('ignore')

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
        
        return {
            'y_train': y_train.values.ravel(),
            'y_test': y_test.values.ravel()
        }
    except FileNotFoundError:
        print(f"[ERROR] Data not found")
        return None

def get_search_space():
    return {
        'use_box_cox': hp.choice('use_box_cox', [True, False]),
        'use_trend': hp.choice('use_trend', [True, False]),
        'use_damped_trend': hp.choice('use_damped_trend', [True, False]),
        'use_arma_errors': hp.choice('use_arma_errors', [True, False])
    }

def train_and_eval(params, data):
    y_train = data['y_train']
    y_test = data['y_test']
    
    try:
        estimator = TBATS(
            use_box_cox=params['use_box_cox'],
            use_trend=params['use_trend'],
            use_damped_trend=params['use_damped_trend'],
            use_arma_errors=params['use_arma_errors'],
            seasonal_periods=[24, 24*365.25]
        )
        model = estimator.fit(y_train)
        preds = model.forecast(steps=len(y_test))

        preds = np.nan_to_num(preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        
        return {'loss': rmse, 'status': STATUS_OK, 'model': model, 'preds': preds}

    except Exception as e:
        print(f"   [WARNING] Trial failed: {e}")
        return {'loss': float('inf'), 'status': STATUS_OK}

def append_results(experiment, best_params, metrics):
    filename = f"{OUTPUT_DIR}/results_tbats.txt"
    header = ""
    if not os.path.exists(filename):
        header = "=" * 50 + "\nMODEL: TBATS\n" + "=" * 50 + "\n\n"
    
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
    print("   TBATS ONLY (WITH DETAILED TRACKING)")
    print("=" * 80)

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
            model_name='tbats',
            experiment_name=experiment,
            y_true=y_test,
            y_pred=preds,
            best_params=best_params,
            additional_info={'mae': metrics['mae'], 'rmse': metrics['rmse'], 'r2': metrics['r2']}
        )
    
    print("\n" + "=" * 80)
    print("TBATS COMPLETE")
    print("=" * 80)

