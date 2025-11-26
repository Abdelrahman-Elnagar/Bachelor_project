"""
03_run_ml_with_detailed.py
---------------------------
ENHANCED VERSION with detailed residual tracking.
Saves per-sample predictions to detailed_results/ directory.
"""

import os
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import warnings

# Import detailed metrics utilities
from utils_detailed_metrics import save_detailed_predictions

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
DATA_DIR = "processed_data"
OUTPUT_DIR = "outputs"
MAX_EVALS = 50

os.makedirs(OUTPUT_DIR, exist_ok=True)

EXPERIMENTS = [
    "with_wsi_solar",
    "no_wsi_solar",
    "with_wsi_wind",
    "no_wsi_wind"
]

MODELS = ["xgboost", "lightgbm", "catboost"]

def load_data(experiment_name):
    """Loads the specific Parquet files for an experiment."""
    try:
        X_train = pd.read_parquet(f"{DATA_DIR}/X_train_{experiment_name}.parquet")
        y_train = pd.read_parquet(f"{DATA_DIR}/y_train_{experiment_name}.parquet").values.ravel()
        X_test = pd.read_parquet(f"{DATA_DIR}/X_test_{experiment_name}.parquet")
        y_test = pd.read_parquet(f"{DATA_DIR}/y_test_{experiment_name}.parquet").values.ravel()
        
        # Filter: Keep ONLY numeric columns
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < len(X_train.columns):
            dropped_cols = set(X_train.columns) - set(numeric_cols)
            print(f"   [INFO] Dropping non-numeric columns: {dropped_cols}")
        
        X_train = X_train[numeric_cols]
        X_test = X_test[numeric_cols]
        
        return X_train, y_train, X_test, y_test
    except FileNotFoundError:
        print(f"[ERROR] Could not find data for {experiment_name}. Run 01_data_forge.py first!")
        return None, None, None, None

def get_aggressive_search_space(model_name):
    """Returns the SOTA Hyperopt search space."""
    if model_name == "xgboost":
        return {
            'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.3)),
            'max_depth': hp.choice('max_depth', range(3, 15)),
            'subsample': hp.uniform('subsample', 0.5, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
            'n_estimators': hp.choice('n_estimators', range(500, 5000, 100)),
            'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-5), np.log(100)),
            'reg_lambda': hp.loguniform('reg_lambda', np.log(1e-5), np.log(100)),
            'min_child_weight': hp.choice('min_child_weight', range(1, 10))
        }

    elif model_name == "lightgbm":
        return {
            'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.3)),
            'num_leaves': hp.choice('num_leaves', range(20, 256)),
            'max_depth': hp.choice('max_depth', [-1, 5, 10, 15, 20, 30]),
            'subsample': hp.uniform('subsample', 0.5, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
            'n_estimators': hp.choice('n_estimators', range(500, 5000, 100)),
            'min_child_samples': hp.choice('min_child_samples', range(5, 100)),
            'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-5), np.log(100)),
            'reg_lambda': hp.loguniform('reg_lambda', np.log(1e-5), np.log(100))
        }

    elif model_name == "catboost":
        return {
            'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.3)),
            'depth': hp.choice('depth', range(4, 11)),
            'l2_leaf_reg': hp.loguniform('l2_leaf_reg', np.log(1), np.log(10)),
            'iterations': hp.choice('iterations', range(500, 3000, 100)),
            'subsample': hp.uniform('subsample', 0.5, 1.0),
            'random_strength': hp.uniform('random_strength', 1, 10),
            'bagging_temperature': hp.uniform('bagging_temperature', 0, 1)
        }
    return {}

def train_and_eval(params, model_name, X_train, y_train, X_test, y_test):
    """Objective function for Hyperopt."""
    
    if model_name == "xgboost":
        model = xgb.XGBRegressor(**params, n_jobs=-1, random_state=42, 
                                 early_stopping_rounds=50)
        model.fit(X_train, y_train, 
                  eval_set=[(X_test, y_test)], 
                  verbose=False)
        
    elif model_name == "lightgbm":
        model = lgb.LGBMRegressor(**params, n_jobs=-1, random_state=42, verbose=-1)
        from lightgbm import early_stopping, log_evaluation
        model.fit(X_train, y_train, 
                  eval_set=[(X_test, y_test)], 
                  callbacks=[early_stopping(stopping_rounds=50), log_evaluation(0)])
        
    elif model_name == "catboost":
        model = cb.CatBoostRegressor(**params, thread_count=-1, random_state=42, 
                                     verbose=0, allow_writing_files=False)
        model.fit(X_train, y_train, 
                  eval_set=(X_test, y_test), 
                  early_stopping_rounds=50, 
                  verbose=False)
    
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    
    return {'loss': rmse, 'status': STATUS_OK, 'model': model}

def append_results_to_file(model_name, experiment, best_params, metrics):
    """Writes the professional report to text file."""
    filename = f"{OUTPUT_DIR}/results_{model_name}.txt"
    
    header = ""
    if not os.path.exists(filename):
        header = f"==================================================\nMODEL: {model_name.upper()} (Aggressive Tuning)\n==================================================\n\n"
    
    report = f"""--- EXPERIMENT: {experiment.upper()} ---
Date: {time.strftime("%Y-%m-%d %H:%M:%S")}
Best Parameters: {best_params}

Performance Metrics:
   MAE : {metrics['mae']:.4f} MW
   RMSE: {metrics['rmse']:.4f} MW
   R2  : {metrics['r2']:.4f}
--------------------------------------------------
"""
    with open(filename, "a") as f:
        f.write(header + report)
    print(f"   [REPORT] Saved to {filename}")

# --- MAIN EXECUTION FLOW ---
if __name__ == "__main__":
    print("========================================")
    print("   PHASE 2: THE MACHINE LEARNING ARENA  ")
    print("   (WITH DETAILED RESIDUAL TRACKING)    ")
    print("========================================")

    for model_name in MODELS:
        print(f"\n>>> ENTERING ARENA: {model_name.upper()} <<<")
        
        for experiment in EXPERIMENTS:
            print(f"\n   [Experiment] {experiment} ...")
            
            # 1. Load Data
            X_train, y_train, X_test, y_test = load_data(experiment)
            if X_train is None: continue
            
            # 2. Define Objective Wrapper
            def objective(params):
                return train_and_eval(params, model_name, X_train, y_train, X_test, y_test)
            
            # 3. Hyperopt Optimization
            print(f"   [Tuning] Running {MAX_EVALS} trials with TPE Bayesian Optimization...")
            space = get_aggressive_search_space(model_name)
            trials = Trials()
            
            best_indices = fmin(fn=objective, 
                                space=space, 
                                algo=tpe.suggest, 
                                max_evals=MAX_EVALS, 
                                trials=trials)
            
            # 4. Retrieve Best Parameters
            best_params = space_eval(space, best_indices)
            print(f"   [Champion Params] {best_params}")
            
            # 5. Retrain Champion
            print("   [Training] Retraining Champion Model for Final Verification...")
            final_run = train_and_eval(best_params, model_name, X_train, y_train, X_test, y_test)
            final_model = final_run['model']
            
            final_preds = final_model.predict(X_test)
            metrics = {
                'mae': mean_absolute_error(y_test, final_preds),
                'rmse': np.sqrt(mean_squared_error(y_test, final_preds)),
                'r2': r2_score(y_test, final_preds)
            }
            
            print(f"   [Result] RMSE: {metrics['rmse']:.2f} | MAE: {metrics['mae']:.2f}")
            
            # 6. Save Standard Report
            append_results_to_file(model_name, experiment, best_params, metrics)
            
            # 7. *** NEW: Save Detailed Predictions and Residuals ***
            save_detailed_predictions(
                model_name=model_name,
                experiment_name=experiment,
                y_true=y_test,
                y_pred=final_preds,
                best_params=best_params,
                additional_info={
                    'mae': metrics['mae'],
                    'rmse': metrics['rmse'],
                    'r2': metrics['r2'],
                    'n_features': X_train.shape[1],
                    'n_train': len(X_train),
                    'n_test': len(X_test)
                }
            )
    
    print("\n" + "=" * 80)
    print("ML TRAINING COMPLETE - Check detailed_results/ for per-sample analysis")
    print("=" * 80)

