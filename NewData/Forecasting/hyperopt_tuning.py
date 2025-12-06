"""
Hyperopt tuning module for all energy prediction models.
Defines search spaces and objective functions for hyperparameter optimization.
"""

import numpy as np
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

from model_wrappers import (
    ProphetWrapper, TBATSWrapper, SARIMAXWrapper,
    XGBoostWrapper, LightGBMWrapper, CatBoostWrapper, RandomForestWrapper
)


def get_prophet_space():
    """Define hyperparameter search space for Prophet."""
    return {
        'changepoint_prior_scale': hp.loguniform('changepoint_prior_scale', np.log(0.001), np.log(0.5)),
        'seasonality_prior_scale': hp.loguniform('seasonality_prior_scale', np.log(0.01), np.log(10)),
        'holidays_prior_scale': hp.loguniform('holidays_prior_scale', np.log(0.01), np.log(10)),
        'seasonality_mode': hp.choice('seasonality_mode', ['additive', 'multiplicative']),
        'yearly_seasonality': hp.choice('yearly_seasonality', [True, False]),
        'weekly_seasonality': hp.choice('weekly_seasonality', [True, False]),
        'daily_seasonality': hp.choice('daily_seasonality', [True, False]),
    }


def get_tbats_space():
    """Define hyperparameter search space for TBATS."""
    return {
        'use_box_cox': hp.choice('use_box_cox', [True, False, None]),
        'use_trend': hp.choice('use_trend', [True, False]),
        'use_damped_trend': hp.choice('use_damped_trend', [True, False]),
        'seasonal_periods': hp.choice('seasonal_periods', [[24], [24, 168], [24, 168, 8760]]),  # hourly, daily, weekly, yearly
        'use_arma_errors': hp.choice('use_arma_errors', [True, False]),
    }


def get_sarimax_space():
    """Define hyperparameter search space for SARIMAX."""
    return {
        'p': hp.choice('p', [0, 1, 2, 3]),
        'd': hp.choice('d', [0, 1, 2]),
        'q': hp.choice('q', [0, 1, 2, 3]),
        'P': hp.choice('P', [0, 1, 2]),
        'D': hp.choice('D', [0, 1]),
        'Q': hp.choice('Q', [0, 1, 2]),
        's': hp.choice('s', [24, 168]),  # hourly or weekly seasonality
    }


def get_xgboost_space():
    """Define hyperparameter search space for XGBoost."""
    return {
        'n_estimators': hp.choice('n_estimators', [100, 200, 300, 500]),
        'max_depth': hp.choice('max_depth', [3, 4, 5, 6, 7, 8]),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
        'subsample': hp.uniform('subsample', 0.6, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
        'reg_alpha': hp.loguniform('reg_alpha', np.log(0.001), np.log(10)),
        'reg_lambda': hp.loguniform('reg_lambda', np.log(0.001), np.log(10)),
        'min_child_weight': hp.choice('min_child_weight', [1, 3, 5, 7]),
    }


def get_lightgbm_space():
    """Define hyperparameter search space for LightGBM."""
    return {
        'n_estimators': hp.choice('n_estimators', [100, 200, 300, 500]),
        'max_depth': hp.choice('max_depth', [3, 4, 5, 6, 7, 8, -1]),  # -1 means no limit
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
        'num_leaves': hp.choice('num_leaves', [15, 31, 50, 70, 100, 127]),
        'subsample': hp.uniform('subsample', 0.6, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
        'reg_alpha': hp.loguniform('reg_alpha', np.log(0.001), np.log(10)),
        'reg_lambda': hp.loguniform('reg_lambda', np.log(0.001), np.log(10)),
        'min_child_samples': hp.choice('min_child_samples', [5, 10, 20, 30]),
    }


def get_catboost_space():
    """Define hyperparameter search space for CatBoost."""
    return {
        'iterations': hp.choice('iterations', [100, 200, 300, 500]),
        'depth': hp.choice('depth', [4, 5, 6, 7, 8, 9, 10]),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
        'l2_leaf_reg': hp.loguniform('l2_leaf_reg', np.log(0.1), np.log(10)),
        'border_count': hp.choice('border_count', [32, 64, 128, 254]),
        'bagging_temperature': hp.uniform('bagging_temperature', 0, 1),
    }


def get_random_forest_space():
    """Define hyperparameter search space for Random Forest."""
    return {
        'n_estimators': hp.choice('n_estimators', [100, 200, 300, 500, 1000]),
        'max_depth': hp.choice('max_depth', [None, 10, 20, 30, 40, 50]),
        'min_samples_split': hp.choice('min_samples_split', [2, 5, 10, 20]),
        'min_samples_leaf': hp.choice('min_samples_leaf', [1, 2, 4, 8]),
        'max_features': hp.choice('max_features', ['sqrt', 'log2', None]),
        'bootstrap': hp.choice('bootstrap', [True, False]),
    }


def prophet_objective(params, X_train, y_train, X_val, y_val):
    """Objective function for Prophet hyperparameter tuning."""
    try:
        wrapper = ProphetWrapper()
        wrapper.fit(X_train, y_train, **params)
        y_pred = wrapper.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val.values, y_pred))
        return {'loss': rmse, 'status': STATUS_OK}
    except Exception as e:
        return {'loss': 1e10, 'status': STATUS_OK}


def tbats_objective(params, X_train, y_train, X_val, y_val):
    """Objective function for TBATS hyperparameter tuning."""
    try:
        wrapper = TBATSWrapper()
        wrapper.fit(X_train, y_train, X_test=X_val, **params)
        y_pred = wrapper.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val.values, y_pred))
        return {'loss': rmse, 'status': STATUS_OK}
    except Exception as e:
        return {'loss': 1e10, 'status': STATUS_OK}


def sarimax_objective(params, X_train, y_train, X_val, y_val):
    """Objective function for SARIMAX hyperparameter tuning."""
    try:
        # Convert to SARIMAX format
        order = (params['p'], params['d'], params['q'])
        seasonal_order = (params['P'], params['D'], params['Q'], params['s'])
        
        wrapper = SARIMAXWrapper()
        wrapper.fit(X_train, y_train, order=order, seasonal_order=seasonal_order)
        y_pred = wrapper.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val.values, y_pred))
        return {'loss': rmse, 'status': STATUS_OK}
    except Exception as e:
        return {'loss': 1e10, 'status': STATUS_OK}


def xgboost_objective(params, X_train, y_train, X_val, y_val):
    """Objective function for XGBoost hyperparameter tuning."""
    try:
        wrapper = XGBoostWrapper()
        wrapper.fit(X_train, y_train, **params)
        y_pred = wrapper.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val.values, y_pred))
        return {'loss': rmse, 'status': STATUS_OK}
    except Exception as e:
        return {'loss': 1e10, 'status': STATUS_OK}


def lightgbm_objective(params, X_train, y_train, X_val, y_val):
    """Objective function for LightGBM hyperparameter tuning."""
    try:
        wrapper = LightGBMWrapper()
        wrapper.fit(X_train, y_train, **params)
        y_pred = wrapper.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val.values, y_pred))
        return {'loss': rmse, 'status': STATUS_OK}
    except Exception as e:
        return {'loss': 1e10, 'status': STATUS_OK}


def catboost_objective(params, X_train, y_train, X_val, y_val):
    """Objective function for CatBoost hyperparameter tuning."""
    try:
        wrapper = CatBoostWrapper()
        wrapper.fit(X_train, y_train, **params)
        y_pred = wrapper.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val.values, y_pred))
        return {'loss': rmse, 'status': STATUS_OK}
    except Exception as e:
        return {'loss': 1e10, 'status': STATUS_OK}


def random_forest_objective(params, X_train, y_train, X_val, y_val):
    """Objective function for Random Forest hyperparameter tuning."""
    try:
        wrapper = RandomForestWrapper()
        wrapper.fit(X_train, y_train, **params)
        y_pred = wrapper.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val.values, y_pred))
        return {'loss': rmse, 'status': STATUS_OK}
    except Exception as e:
        return {'loss': 1e10, 'status': STATUS_OK}


def tune_hyperparameters(model_name, X_train, y_train, X_val, y_val, max_evals=100):
    """
    Tune hyperparameters for a given model.
    
    Args:
        model_name: Name of the model to tune
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        max_evals: Maximum number of hyperparameter evaluations
        
    Returns:
        best_params: Best hyperparameters found
        best_model: Trained model with best parameters
        best_rmse: Best RMSE achieved
    """
    # Get search space and objective function
    model_configs = {
        'Prophet': {
            'space': get_prophet_space(),
            'objective': prophet_objective,
            'max_evals': min(max_evals, 50)  # Statistical models need fewer evals
        },
        'TBATS': {
            'space': get_tbats_space(),
            'objective': tbats_objective,
            'max_evals': min(max_evals, 50)
        },
        'SARIMAX': {
            'space': get_sarimax_space(),
            'objective': sarimax_objective,
            'max_evals': min(max_evals, 50)
        },
        'XGBoost': {
            'space': get_xgboost_space(),
            'objective': xgboost_objective,
            'max_evals': max_evals
        },
        'LightGBM': {
            'space': get_lightgbm_space(),
            'objective': lightgbm_objective,
            'max_evals': max_evals
        },
        'CatBoost': {
            'space': get_catboost_space(),
            'objective': catboost_objective,
            'max_evals': max_evals
        },
        'RandomForest': {
            'space': get_random_forest_space(),
            'objective': random_forest_objective,
            'max_evals': max_evals
        }
    }
    
    if model_name not in model_configs:
        raise ValueError(f"Unknown model: {model_name}")
    
    config = model_configs[model_name]
    
    # Create trials object
    trials = Trials()
    
    # Define objective function with data
    def objective(params):
        return config['objective'](params, X_train, y_train, X_val, y_val)
    
    # Run hyperparameter optimization
    best = fmin(
        fn=objective,
        space=config['space'],
        algo=tpe.suggest,
        max_evals=config['max_evals'],
        trials=trials,
        verbose=False
    )
    
    # Get best parameters using space_eval to convert Apply objects to actual values
    best_params = space_eval(config['space'], best)
    
    # Train final model with best parameters
    wrapper_classes = {
        'Prophet': ProphetWrapper,
        'TBATS': TBATSWrapper,
        'SARIMAX': SARIMAXWrapper,
        'XGBoost': XGBoostWrapper,
        'LightGBM': LightGBMWrapper,
        'CatBoost': CatBoostWrapper,
        'RandomForest': RandomForestWrapper
    }
    
    wrapper_class = wrapper_classes[model_name]
    best_model = wrapper_class()
    
    # Handle SARIMAX special case
    if model_name == 'SARIMAX':
        order = (best_params['p'], best_params['d'], best_params['q'])
        seasonal_order = (best_params['P'], best_params['D'], best_params['Q'], best_params['s'])
        best_model.fit(X_train, y_train, order=order, seasonal_order=seasonal_order)
        best_params = {'order': order, 'seasonal_order': seasonal_order}
    else:
        best_model.fit(X_train, y_train, **best_params)
    
    # Get best RMSE
    best_rmse = min([t['result']['loss'] for t in trials.trials])
    
    return best_params, best_model, best_rmse
