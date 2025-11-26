"""
enhanced_05b_lear.py
--------------------
LEAR (Lasso Elastic-net AutoRegressive) ONLY - WITH DETAILED RESIDUAL TRACKING
"""

import os
import pandas as pd
import numpy as np
import time
import warnings
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils_detailed_metrics import save_detailed_predictions

warnings.filterwarnings('ignore')

DATA_DIR = "../processed_data"
OUTPUT_DIR = "../outputs"
MAX_EVALS = 20

os.makedirs(OUTPUT_DIR, exist_ok=True)

EXPERIMENTS = ["with_wsi_solar", "no_wsi_solar", "with_wsi_wind", "no_wsi_wind"]

def load_data_raw(experiment_name):
    try:
        X_train_df = pd.read_parquet(f"{DATA_DIR}/X_train_{experiment_name}.parquet")
        X_test_df = pd.read_parquet(f"{DATA_DIR}/X_test_{experiment_name}.parquet")
        
        X_train = X_train_df.select_dtypes(include=[np.number]).values
        X_test = X_test_df.select_dtypes(include=[np.number]).values
        
        y_train = pd.read_parquet(f"{DATA_DIR}/y_train_{experiment_name}.parquet").values.ravel()
        y_test = pd.read_parquet(f"{DATA_DIR}/y_test_{experiment_name}.parquet").values.ravel()
        
        return X_train, y_train, X_test, y_test
    except FileNotFoundError:
        print(f"[ERROR] Data not found")
        return None, None, None, None

def create_lags(X, y, lags):
    X_sliced = X[lags:].copy()
    
    lag_features = []
    for i in range(1, lags + 1):
        lag_features.append(y[lags-i : -i])
    
    if not lag_features:
        return X, y
        
    lag_matrix = np.column_stack(lag_features)
    X_final = np.hstack([X_sliced, lag_matrix])
    y_final = y[lags:]
    
    return X_final, y_final

def get_search_space():
    return {
        'alpha': hp.loguniform('alpha', np.log(0.0001), np.log(1.0)),
        'l1_ratio': hp.uniform('l1_ratio', 0.0, 1.0),
        'lags': hp.choice('lags', [24, 48, 168])
    }

def train_and_eval(params, X_train, y_train, X_test, y_test):
    lags = params['lags']
    X_tr_lag, y_tr_lag = create_lags(X_train, y_train, lags)
    X_te_lag, y_te_lag = create_lags(X_test, y_test, lags)
    
    model = ElasticNet(alpha=params['alpha'], l1_ratio=params['l1_ratio'], random_state=42, max_iter=2000)
    model.fit(X_tr_lag, y_tr_lag)
    
    preds = model.predict(X_te_lag)
    rmse = np.sqrt(mean_squared_error(y_te_lag, preds))
    return {'loss': rmse, 'status': STATUS_OK, 'model': model, 'lags': lags}

def append_results(experiment, best_params, metrics):
    filename = f"{OUTPUT_DIR}/results_lear.txt"
    header = ""
    if not os.path.exists(filename):
        header = "=" * 50 + "\nMODEL: LEAR\n" + "=" * 50 + "\n\n"
    
    report = f"""--- EXPERIMENT: {experiment.upper()} ---
Date: {time.strftime("%Y-%m-%d %H:%M:%S")}
Best Parameters: {best_params}
MAE : {metrics['mae']:.4f}
RMSE: {metrics['rmse']:.4f}
R2  : {metrics['r2']:.4f}
{'-' * 50}
"""
    with open(filename, "a") as f:
        f.write(header + report)
    print(f"   [REPORT] Saved to {filename}")

if __name__ == "__main__":
    print("=" * 80)
    print("   LEAR ONLY (WITH DETAILED TRACKING)")
    print("=" * 80)

    for experiment in EXPERIMENTS:
        print(f"\n>>> EXPERIMENT: {experiment.upper()} <<<")
        
        X_train, y_train, X_test, y_test = load_data_raw(experiment)
        if X_train is None: continue
        
        def objective(params):
            return train_and_eval(params, X_train, y_train, X_test, y_test)
        
        print(f"   [Tuning] Running {MAX_EVALS} trials...")
        space = get_search_space()
        trials = Trials()
        best_indices = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=MAX_EVALS, trials=trials)
        best_params = space_eval(space, best_indices)
        
        print("   [Training] Final Run...")
        res = train_and_eval(best_params, X_train, y_train, X_test, y_test)
        lags = res['lags']
        _, y_true = create_lags(X_test, y_test, lags)
        
        model = res['model']
        X_te_lag, _ = create_lags(X_test, y_test, lags)
        y_pred = model.predict(X_te_lag)

        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred)
        }
        print(f"   [Result] RMSE: {metrics['rmse']:.2f} | MAE: {metrics['mae']:.2f}")
        
        append_results(experiment, best_params, metrics)
        
        save_detailed_predictions(
            model_name='lear',
            experiment_name=experiment,
            y_true=y_true,
            y_pred=y_pred,
            best_params=best_params,
            additional_info={'mae': metrics['mae'], 'rmse': metrics['rmse'], 'r2': metrics['r2']}
        )
    
    print("\n" + "=" * 80)
    print("LEAR COMPLETE")
    print("=" * 80)

