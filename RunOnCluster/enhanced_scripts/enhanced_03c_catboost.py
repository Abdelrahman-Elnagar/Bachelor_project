"""
enhanced_03c_catboost.py
------------------------
CATBOOST ONLY - WITH DETAILED RESIDUAL TRACKING
"""

import os
import pandas as pd
import numpy as np
import catboost as cb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import warnings
from utils_detailed_metrics import save_detailed_predictions

warnings.filterwarnings('ignore')

DATA_DIR = "../processed_data"
OUTPUT_DIR = "../outputs"
MAX_EVALS = 50

os.makedirs(OUTPUT_DIR, exist_ok=True)

EXPERIMENTS = ["with_wsi_solar", "no_wsi_solar", "with_wsi_wind", "no_wsi_wind"]

def load_data(experiment_name):
    try:
        X_train = pd.read_parquet(f"{DATA_DIR}/X_train_{experiment_name}.parquet")
        y_train = pd.read_parquet(f"{DATA_DIR}/y_train_{experiment_name}.parquet").values.ravel()
        X_test = pd.read_parquet(f"{DATA_DIR}/X_test_{experiment_name}.parquet")
        y_test = pd.read_parquet(f"{DATA_DIR}/y_test_{experiment_name}.parquet").values.ravel()
        
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        X_train = X_train[numeric_cols]
        X_test = X_test[numeric_cols]
        
        return X_train, y_train, X_test, y_test
    except FileNotFoundError:
        print(f"[ERROR] Could not find data for {experiment_name}")
        return None, None, None, None

def get_search_space():
    return {
        'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.3)),
        'depth': hp.choice('depth', range(4, 11)),
        'l2_leaf_reg': hp.loguniform('l2_leaf_reg', np.log(1), np.log(10)),
        'iterations': hp.choice('iterations', range(500, 3000, 100)),
        'subsample': hp.uniform('subsample', 0.5, 1.0),
        'random_strength': hp.uniform('random_strength', 1, 10),
        'bagging_temperature': hp.uniform('bagging_temperature', 0, 1)
    }

def train_and_eval(params, X_train, y_train, X_test, y_test):
    model = cb.CatBoostRegressor(**params, thread_count=-1, random_state=42, 
                                 verbose=0, allow_writing_files=False)
    model.fit(X_train, y_train, 
              eval_set=(X_test, y_test), 
              early_stopping_rounds=50, 
              verbose=False)
    
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    
    return {'loss': rmse, 'status': STATUS_OK, 'model': model}

def append_results(experiment, best_params, metrics):
    filename = f"{OUTPUT_DIR}/results_catboost.txt"
    header = ""
    if not os.path.exists(filename):
        header = "=" * 50 + "\nMODEL: CATBOOST (Aggressive Tuning)\n" + "=" * 50 + "\n\n"
    
    report = f"""--- EXPERIMENT: {experiment.upper()} ---
Date: {time.strftime("%Y-%m-%d %H:%M:%S")}
Best Parameters: {best_params}

Performance Metrics:
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
    print("   CATBOOST ONLY (WITH DETAILED TRACKING)")
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
        
        best_indices = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=MAX_EVALS, trials=trials)
        best_params = space_eval(space, best_indices)
        print(f"   [Champion Params] {best_params}")
        
        print("   [Training] Final Run...")
        final_run = train_and_eval(best_params, X_train, y_train, X_test, y_test)
        final_model = final_run['model']
        
        final_preds = final_model.predict(X_test)
        metrics = {
            'mae': mean_absolute_error(y_test, final_preds),
            'rmse': np.sqrt(mean_squared_error(y_test, final_preds)),
            'r2': r2_score(y_test, final_preds)
        }
        
        print(f"   [Result] RMSE: {metrics['rmse']:.2f} | MAE: {metrics['mae']:.2f}")
        
        append_results(experiment, best_params, metrics)
        
        save_detailed_predictions(
            model_name='catboost',
            experiment_name=experiment,
            y_true=y_test,
            y_pred=final_preds,
            best_params=best_params,
            additional_info={'mae': metrics['mae'], 'rmse': metrics['rmse'], 'r2': metrics['r2']}
        )
    
    print("\n" + "=" * 80)
    print("CATBOOST COMPLETE")
    print("=" * 80)

