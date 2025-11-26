"""
generate_individual_scripts.py
-------------------------------
Generates individual scripts for each model (02a, 02b, 03a, 03b, etc.)
Run this once to create all individual model scripts.
"""

import os

# Model configurations
MODELS_CONFIG = {
    # Statistical Models (02)
    '02a_prophet': {
        'model': 'Prophet',
        'import': 'from prophet import Prophet',
        'search_space': """return {
        'changepoint_prior_scale': hp.loguniform('changepoint_prior_scale', np.log(0.001), np.log(0.5)),
        'seasonality_prior_scale': hp.loguniform('seasonality_prior_scale', np.log(0.01), np.log(10.0)),
        'seasonality_mode': hp.choice('seasonality_mode', ['additive', 'multiplicative']),
        'changepoint_range': hp.uniform('changepoint_range', 0.8, 0.95)
    }""",
        'train': """df_train = pd.DataFrame({'ds': data['train_ds'], 'y': y_train})
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
        preds = np.nan_to_num(preds)""",
        'data_loader': 'stat',
        'max_evals': 30
    },
    
    # ML Models (03)
    '03a_xgboost': {
        'model': 'XGBoost',
        'import': 'import xgboost as xgb',
        'search_space': """return {
        'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.3)),
        'max_depth': hp.choice('max_depth', range(3, 15)),
        'subsample': hp.uniform('subsample', 0.5, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
        'n_estimators': hp.choice('n_estimators', range(500, 5000, 100)),
        'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-5), np.log(100)),
        'reg_lambda': hp.loguniform('reg_lambda', np.log(1e-5), np.log(100)),
        'min_child_weight': hp.choice('min_child_weight', range(1, 10))
    }""",
        'train': """model = xgb.XGBRegressor(**params, n_jobs=-1, random_state=42, early_stopping_rounds=50)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        preds = model.predict(X_test)""",
        'data_loader': 'ml',
        'max_evals': 50
    },
    
    '03b_lightgbm': {
        'model': 'LightGBM',
        'import': 'import lightgbm as lgb\nfrom lightgbm import early_stopping, log_evaluation',
        'search_space': """return {
        'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.3)),
        'num_leaves': hp.choice('num_leaves', range(20, 256)),
        'max_depth': hp.choice('max_depth', [-1, 5, 10, 15, 20, 30]),
        'subsample': hp.uniform('subsample', 0.5, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
        'n_estimators': hp.choice('n_estimators', range(500, 5000, 100)),
        'min_child_samples': hp.choice('min_child_samples', range(5, 100)),
        'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-5), np.log(100)),
        'reg_lambda': hp.loguniform('reg_lambda', np.log(1e-5), np.log(100))
    }""",
        'train': """model = lgb.LGBMRegressor(**params, n_jobs=-1, random_state=42, verbose=-1)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], 
                  callbacks=[early_stopping(stopping_rounds=50), log_evaluation(0)])
        preds = model.predict(X_test)""",
        'data_loader': 'ml',
        'max_evals': 50
    },
    
    '03c_catboost': {
        'model': 'CatBoost',
        'import': 'import catboost as cb',
        'search_space': """return {
        'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.3)),
        'depth': hp.choice('depth', range(4, 11)),
        'l2_leaf_reg': hp.loguniform('l2_leaf_reg', np.log(1), np.log(10)),
        'iterations': hp.choice('iterations', range(500, 3000, 100)),
        'subsample': hp.uniform('subsample', 0.5, 1.0),
        'random_strength': hp.uniform('random_strength', 1, 10),
        'bagging_temperature': hp.uniform('bagging_temperature', 0, 1)
    }""",
        'train': """model = cb.CatBoostRegressor(**params, thread_count=-1, random_state=42, 
                                     verbose=0, allow_writing_files=False)
        model.fit(X_train, y_train, eval_set=(X_test, y_test), 
                  early_stopping_rounds=50, verbose=False)
        preds = model.predict(X_test)""",
        'data_loader': 'ml',
        'max_evals': 50
    },
}

# Template for individual model scripts
SCRIPT_TEMPLATE = '''"""
enhanced_{filename}.py
{description}
{model_name} ONLY - WITH DETAILED RESIDUAL TRACKING
"""

import os
import pandas as pd
import numpy as np
import time
import warnings
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
{model_import}
from utils_detailed_metrics import save_detailed_predictions

warnings.filterwarnings('ignore')

DATA_DIR = "../processed_data"
OUTPUT_DIR = "../outputs"
MAX_EVALS = {max_evals}

os.makedirs(OUTPUT_DIR, exist_ok=True)

EXPERIMENTS = ["with_wsi_solar", "no_wsi_solar", "with_wsi_wind", "no_wsi_wind"]

{data_loader_func}

def get_search_space():
    {search_space}

def train_and_eval(params, {train_args}):
    try:
        {train_code}
        
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        return {{'loss': rmse, 'status': STATUS_OK, 'preds': preds}}
    except Exception as e:
        print(f"   [WARNING] Trial failed: {{e}}")
        return {{'loss': float('inf'), 'status': STATUS_OK}}

def append_results(experiment, best_params, metrics):
    filename = f"{{OUTPUT_DIR}}/results_{model_lower}.txt"
    header = ""
    if not os.path.exists(filename):
        header = "=" * 50 + "\\nMODEL: {model_upper}\\n" + "=" * 50 + "\\n\\n"
    
    report = f"""--- EXPERIMENT: {{experiment.upper()}} ---
Date: {{time.strftime("%Y-%m-%d %H:%M:%S")}}
Best Parameters: {{best_params}}
MAE : {{metrics['mae']:.4f}} MW
RMSE: {{metrics['rmse']:.4f}} MW
R2  : {{metrics['r2']:.4f}}
{'-' * 50}
"""
    with open(filename, "a") as f:
        f.write(header + report)
    print(f"   [REPORT] Saved to {{filename}}")

if __name__ == "__main__":
    print("=" * 80)
    print("   {model_upper} ONLY (WITH DETAILED TRACKING)")
    print("=" * 80)

    for experiment in EXPERIMENTS:
        print(f"\\n>>> EXPERIMENT: {{experiment.upper()}} <<<")
        
        {load_data_call}
        if {data_check}: continue
        
        def objective(params):
            return train_and_eval(params, {objective_args})
        
        print(f"   [Tuning] Running {{MAX_EVALS}} trials...")
        space = get_search_space()
        trials = Trials()
        
        best_indices = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=MAX_EVALS, trials=trials)
        best_params = space_eval(space, best_indices)
        
        print("   [Training] Final Run...")
        final_res = train_and_eval(best_params, {objective_args})
        preds = final_res['preds']
        y_test = {y_test_var}
        
        metrics = {{
            'mae': mean_absolute_error(y_test, preds),
            'rmse': np.sqrt(mean_squared_error(y_test, preds)),
            'r2': r2_score(y_test, preds)
        }}
        print(f"   [Result] RMSE: {{metrics['rmse']:.2f}} | MAE: {{metrics['mae']:.2f}}")
        
        append_results(experiment, best_params, metrics)
        
        save_detailed_predictions(
            model_name='{model_lower}',
            experiment_name=experiment,
            y_true=y_test,
            y_pred=preds,
            best_params=best_params,
            additional_info={{'mae': metrics['mae'], 'rmse': metrics['rmse'], 'r2': metrics['r2']}}
        )
    
    print("\\n" + "=" * 80)
    print("{model_upper} COMPLETE")
    print("=" * 80)
'''

# Data loader templates
DATA_LOADER_ML = '''def load_data(experiment_name):
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
        print(f"[ERROR] Could not find data")
        return None, None, None, None'''

DATA_LOADER_STAT = '''def load_and_adapt_data(experiment_name):
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
        print(f"[ERROR] Data not found")
        return None'''

def generate_script(filename, config):
    """Generate a single model script."""
    
    model_lower = config['model'].lower().replace(' ', '_')
    model_upper = config['model'].upper()
    
    # Select data loader
    if config['data_loader'] == 'ml':
        data_loader_func = DATA_LOADER_ML
        load_data_call = "X_train, y_train, X_test, y_test = load_data(experiment)"
        data_check = "X_train is None"
        train_args = "X_train, y_train, X_test, y_test"
        objective_args = "X_train, y_train, X_test, y_test"
        y_test_var = "y_test"
    else:  # stat
        data_loader_func = DATA_LOADER_STAT
        load_data_call = "data = load_and_adapt_data(experiment)"
        data_check = "data is None"
        train_args = "data"
        objective_args = "data"
        y_test_var = "data['y_test']"
    
    # Fill template
    script_content = SCRIPT_TEMPLATE.format(
        filename=filename,
        description="-" * (len(filename) + 4),
        model_name=config['model'],
        model_import=config['import'],
        max_evals=config['max_evals'],
        data_loader_func=data_loader_func,
        search_space=config['search_space'],
        train_args=train_args,
        train_code=config['train'],
        model_lower=model_lower,
        model_upper=model_upper,
        load_data_call=load_data_call,
        data_check=data_check,
        objective_args=objective_args,
        y_test_var=y_test_var
    )
    
    output_file = f"enhanced_{filename}.py"
    with open(output_file, 'w') as f:
        f.write(script_content)
    
    print(f"✅ Created: {output_file}")

if __name__ == "__main__":
    print("=" * 80)
    print("GENERATING INDIVIDUAL MODEL SCRIPTS")
    print("=" * 80)
    
    for filename, config in MODELS_CONFIG.items():
        generate_script(filename, config)
    
    print("\n" + "=" * 80)
    print(f"Generated {len(MODELS_CONFIG)} individual model scripts!")
    print("=" * 80)

