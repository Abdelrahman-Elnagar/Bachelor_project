"""
02_run_stat.py
----------------
THE STATISTICAL GUARDIANS.
Models: Prophet, TBATS, SARIMAX.
Logic:
  1. Loads Parquet data.
  2. ADAPTS data: Reconstructs timestamps for Prophet/TBATS (Critical step).
  3. Uses Hyperopt to tune statistical hyperparameters.
  4. Saves results to outputs/.
"""

import os
import pandas as pd
import numpy as np
import time
import warnings
import logging
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Model Imports
from prophet import Prophet
from tbats import TBATS
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

# --- CONFIGURATION ---
DATA_DIR = "processed_data"
OUTPUT_DIR = "outputs"
MAX_EVALS = 30  # Statistical models are slower; 30 is a good balance for SOTA

os.makedirs(OUTPUT_DIR, exist_ok=True)

EXPERIMENTS = [
    "with_wsi_solar",
    "no_wsi_solar",
    "with_wsi_wind",
    "no_wsi_wind"
]

MODELS = ["prophet", "tbats", "sarimax"]

# --- DATA ADAPTERS ---
def load_and_adapt_data(experiment_name):
    """
    Loads parquet and reconstructs format for Stats models.
    Prophet needs: ['ds', 'y'] columns.
    TBATS/SARIMAX need: pandas Series with DatetimeIndex.
    """
    try:
        X_train = pd.read_parquet(f"{DATA_DIR}/X_train_{experiment_name}.parquet")
        y_train = pd.read_parquet(f"{DATA_DIR}/y_train_{experiment_name}.parquet")
        X_test = pd.read_parquet(f"{DATA_DIR}/X_test_{experiment_name}.parquet")
        y_test = pd.read_parquet(f"{DATA_DIR}/y_test_{experiment_name}.parquet")
        
        # --- TIMESTAMPS RECOVERY ---
        # We assume the data is hourly (standard for energy).
        # Since 01_data_forge dropped the index, we recreate a dummy hourly index 
        # relative to a standard start time (e.g. 2022-01-01). 
        # Ideally, X_train has a 'timestamp' column. If not, we generate one.
        
        start_date = "2022-01-01 00:00:00" # Arbitrary start if real one missing
        
        # Check if 'timestamp' or similar exists in columns
        time_col = None
        for col in X_train.columns:
            if 'time' in col.lower() or 'date' in col.lower():
                time_col = col
                break
        
        if time_col:
            # Use existing time column
            train_dates = pd.to_datetime(X_train[time_col])
            test_dates = pd.to_datetime(X_test[time_col])
            # Drop time col from features for SARIMAX (it can't handle dates as input)
            X_train_exog = X_train.drop(columns=[time_col])
            X_test_exog = X_test.drop(columns=[time_col])
        else:
            # Generate synthetic hourly index
            train_dates = pd.date_range(start=start_date, periods=len(y_train), freq='H')
            test_dates = pd.date_range(start=train_dates[-1], periods=len(y_test) + 1, freq='H')[1:]
            X_train_exog = X_train
            X_test_exog = X_test
        
        # Safety: Keep ONLY numeric columns (statistical models need numeric features)
        X_train_exog = X_train_exog.select_dtypes(include=[np.number])
        X_test_exog = X_test_exog.select_dtypes(include=[np.number])

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

def get_search_space(model_name):
    """Aggressive Search Spaces for Statistical Models"""
    
    if model_name == "prophet":
        return {
            'changepoint_prior_scale': hp.loguniform('changepoint_prior_scale', np.log(0.001), np.log(0.5)),
            'seasonality_prior_scale': hp.loguniform('seasonality_prior_scale', np.log(0.01), np.log(10.0)),
            'seasonality_mode': hp.choice('seasonality_mode', ['additive', 'multiplicative']),
            # How much of history to use for trend learning (80% to 95%)
            'changepoint_range': hp.uniform('changepoint_range', 0.8, 0.95)
        }
    
    elif model_name == "tbats":
        return {
            # TBATS auto-tunes well, but we can force structural choices
            'use_box_cox': hp.choice('use_box_cox', [True, False]),
            'use_trend': hp.choice('use_trend', [True, False]),
            'use_damped_trend': hp.choice('use_damped_trend', [True, False]),
            'use_arma_errors': hp.choice('use_arma_errors', [True, False])
        }
        
    elif model_name == "sarimax":
        return {
            # We search for (p,d,q) orders. 
            # WARNING: High orders make training very slow.
            'p': hp.choice('p', [0, 1, 2]),
            'd': hp.choice('d', [0, 1]),
            'q': hp.choice('q', [0, 1, 2]),
            'trend': hp.choice('trend', ['n', 'c', 't', 'ct'])
            # Note: Seasonality (P,D,Q,s) is omitted here to save compute time, 
            # but can be added if you have 24h seasonality explicitly known.
        }
    return {}

def train_and_eval(params, model_name, data):
    """Objective Function"""
    y_train = data['y_train']
    y_test = data['y_test']
    X_train = data['X_train']
    X_test = data['X_test']
    
    model = None
    preds = None
    
    try:
        if model_name == "prophet":
            # 1. Prepare DataFrame for Prophet
            df_train = pd.DataFrame({'ds': data['train_ds'], 'y': y_train})
            
            # Add regressors (Exogenous variables)
            # Prophet expects regressors as columns in the main df
            for col in X_train.columns:
                df_train[col] = X_train[col].values
            
            # Initialize
            m = Prophet(**params)
            
            # Add regressors to model config
            for col in X_train.columns:
                m.add_regressor(col)
                
            m.fit(df_train)
            
            # Make Future DataFrame
            future = pd.DataFrame({'ds': data['test_ds']})
            for col in X_test.columns:
                future[col] = X_test[col].values
                
            forecast = m.predict(future)
            preds = forecast['yhat'].values
            model = m

        elif model_name == "tbats":
            # TBATS handles complex seasonality well but is slow with exogenous vars
            # Note: TBATS python lib support for exogenous (regressors) is limited compared to R.
            # We fit on Univariate y_train mostly, or use simple settings.
            
            estimator = TBATS(
                use_box_cox=params['use_box_cox'],
                use_trend=params['use_trend'],
                use_damped_trend=params['use_damped_trend'],
                use_arma_errors=params['use_arma_errors'],
                seasonal_periods=[24, 24*365.25] # Daily and Yearly seasonality
            )
            model = estimator.fit(y_train)
            preds = model.forecast(steps=len(y_test))

        elif model_name == "sarimax":
            # SARIMAX with Exogenous variables
            order = (params['p'], params['d'], params['q'])
            
            # Enforce stationarity checks
            model = SARIMAX(y_train, exog=X_train, order=order, trend=params['trend'],
                            enforce_stationarity=False, enforce_invertibility=False)
            model_fit = model.fit(disp=False)
            
            preds = model_fit.forecast(steps=len(y_test), exog=X_test)
            model = model_fit

        # --- EVALUATION ---
        # Handle NaNs in predictions (common in Prophet/SARIMAX failures)
        preds = np.nan_to_num(preds)
        
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        
        return {'loss': rmse, 'status': STATUS_OK, 'model': model, 'preds': preds}

    except Exception as e:
        print(f"   [WARNING] Trial failed: {e}")
        return {'loss': float('inf'), 'status': STATUS_OK}

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
    print("   PHASE 2: THE STATISTICAL ARENA       ")
    print("   (Prophet / TBATS / SARIMAX)          ")
    print("========================================")

    for model_name in MODELS:
        print(f"\n>>> ENTERING ARENA: {model_name.upper()} <<<")
        
        for experiment in EXPERIMENTS:
            print(f"\n   [Experiment] {experiment} ...")
            
            data = load_and_adapt_data(experiment)
            if data is None: continue
            
            # Define Objective Wrapper
            def objective(params):
                return train_and_eval(params, model_name, data)
            
            # Hyperopt Optimization
            print(f"   [Tuning] Running {MAX_EVALS} trials...")
            space = get_search_space(model_name)
            trials = Trials()
            
            best_indices = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=MAX_EVALS, trials=trials)
            best_params = space_eval(space, best_indices)
            
            # Final Run
            print("   [Training] Final Run with Champion Parameters...")
            final_res = train_and_eval(best_params, model_name, data)
            preds = final_res['preds']
            y_test = data['y_test']
            
            metrics = {
                'mae': mean_absolute_error(y_test, preds),
                'rmse': np.sqrt(mean_squared_error(y_test, preds)),
                'r2': r2_score(y_test, preds)
            }
            print(f"   [Result] RMSE: {metrics['rmse']:.2f}")
            append_results_to_file(model_name, experiment, best_params, metrics)