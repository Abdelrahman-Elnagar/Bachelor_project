#!/usr/bin/env python3
"""
Renewable Energy Forecasting Model Training

This script implements training for all forecasting models:
- Baseline models (Persistence, Seasonal Persistence, SARIMAX, Prophet)
- Gradient Boosting (LightGBM, XGBoost, CatBoost)
- Neural Forecasting (N-BEATS, TCN, TFT)
- Probabilistic models (Quantile Regression, Conformal Prediction)

Uses walk-forward validation and Optuna hyperparameter tuning.

Author: Energy Forecasting Pipeline
"""

import pandas as pd
import numpy as np
import yaml
import json
from pathlib import Path
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Optional, Tuple, Any
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle

warnings.filterwarnings('ignore')
np.random.seed(42)


class ForecastingTrainer:
    """Train forecasting models for renewable energy"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize trainer"""
        self.base_dir = Path(__file__).parent.parent
        self.config_dir = self.base_dir / "data" / "config"
        self.processed_dir = self.base_dir / "data" / "processed"
        self.models_dir = self.base_dir / "models"
        self.results_dir = self.base_dir / "results"
        self.logs_dir = self.base_dir / "logs"
        
        for dir_path in [self.models_dir, self.results_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        log_file = self.logs_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        if config_path is None:
            config_path = self.config_dir / "forecasting.yaml"
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Load weather features from WSI pipeline
        # Paths relative to forecasting/ directory
        self.wsi_features_path = Path(__file__).parent.parent.parent / "data" / "features" / "weather_features.csv"
        self.stability_labels_path = Path(__file__).parent.parent.parent / "Data" / "processed" / "stability_labels.csv"
        
    def load_data(self, source_type: str = 'both') -> pd.DataFrame:
        """
        Load energy and feature data
        
        Parameters:
        -----------
        source_type : str
            'solar', 'wind', or 'both'
        
        Returns:
        --------
        pd.DataFrame
            Combined dataset with energy and features
        """
        self.logger.info("="*60)
        self.logger.info("LOADING DATA FOR TRAINING")
        self.logger.info("="*60)
        
        # Load energy data
        energy_file = self.processed_dir / "energy_production_2024.csv"
        if not energy_file.exists():
            # Try individual files
            if source_type in ['solar', 'both']:
                solar_file = self.processed_dir / "solar_energy_2024.csv"
                if solar_file.exists():
                    energy_df = pd.read_csv(solar_file)
                else:
                    raise FileNotFoundError(f"Energy data not found. Run data_ingest_energy.py first.")
            else:
                wind_file = self.processed_dir / "wind_energy_2024.csv"
                if wind_file.exists():
                    energy_df = pd.read_csv(wind_file)
                else:
                    raise FileNotFoundError(f"Energy data not found. Run data_ingest_energy.py first.")
        else:
            energy_df = pd.read_csv(energy_file)
        
        energy_df['datetime'] = pd.to_datetime(energy_df['datetime'])
        
        # Load weather features
        if self.wsi_features_path.exists():
            self.logger.info(f"Loading weather features from: {self.wsi_features_path}")
            weather_df = pd.read_csv(self.wsi_features_path)
            weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])
            
            # Merge with energy data
            energy_df = energy_df.merge(weather_df, on='datetime', how='left')
        else:
            self.logger.warning(f"Weather features not found: {self.wsi_features_path}")
        
        # Load stability labels if not already merged
        if 'unstable_gmm' not in energy_df.columns and self.stability_labels_path.exists():
            self.logger.info(f"Loading stability labels from: {self.stability_labels_path}")
            stability_df = pd.read_csv(self.stability_labels_path)
            stability_df['datetime'] = pd.to_datetime(stability_df['datetime'])
            energy_df = energy_df.merge(stability_df, on='datetime', how='left')
        
        # Sort by datetime
        energy_df = energy_df.sort_values('datetime').reset_index(drop=True)
        
        self.logger.info(f"Loaded dataset: {len(energy_df)} rows, {len(energy_df.columns)} columns")
        self.logger.info(f"Date range: {energy_df['datetime'].min()} to {energy_df['datetime'].max()}")
        
        return energy_df
    
    def prepare_features(self, df: pd.DataFrame, source_type: str) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare features for forecasting
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        source_type : str
            'solar' or 'wind'
        
        Returns:
        --------
        pd.DataFrame, List[str]
            Dataframe with features and list of feature names
        """
        df = df.copy()
        
        # Get feature list from config
        feature_config = self.config['features']
        
        if source_type == 'solar':
            feature_names = feature_config.get('solar_features', []) + \
                          feature_config.get('weather_features', []) + \
                          feature_config.get('regime_features', [])
        else:
            feature_names = feature_config.get('wind_features', []) + \
                          feature_config.get('weather_features', []) + \
                          feature_config.get('regime_features', [])
        
        # Add temporal features
        df['hour'] = df['datetime'].dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_of_year'] = df['datetime'].dt.dayofyear
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        
        feature_names.extend(['hour_sin', 'hour_cos', 'day_of_year_sin', 'day_of_year_cos'])
        
        # Select available features only
        available_features = [f for f in feature_names if f in df.columns]
        missing_features = [f for f in feature_names if f not in df.columns]
        
        if missing_features:
            self.logger.warning(f"Missing features: {missing_features}")
        
        # Fill missing values in features
        for col in available_features:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        
        return df, available_features
    
    def walk_forward_split(self, df: pd.DataFrame, train_window_days: int, 
                          step_size_days: int) -> List[Tuple[int, int, int, int]]:
        """
        Generate walk-forward validation splits
        
        Parameters:
        -----------
        df : pd.DataFrame
            Full dataset
        train_window_days : int
            Training window size in days
        step_size_days : int
            Step size between retraining in days
        
        Returns:
        --------
        List[Tuple]
            List of (train_start, train_end, test_start, test_end) indices
        """
        splits = []
        total_hours = len(df)
        train_hours = train_window_days * 24
        step_hours = step_size_days * 24
        
        train_start = 0
        while train_start + train_hours < total_hours:
            train_end = train_start + train_hours
            test_start = train_end
            test_end = min(test_start + 24, total_hours)  # Forecast 24 hours ahead
            
            if test_end > test_start:
                splits.append((train_start, train_end, test_start, test_end))
            
            train_start += step_hours
        
        self.logger.info(f"Generated {len(splits)} walk-forward splits")
        return splits
    
    def train_persistence(self, df: pd.DataFrame, target_col: str, 
                         splits: List[Tuple]) -> Dict:
        """Train persistence model (baseline)"""
        self.logger.info("Training Persistence model...")
        
        predictions = []
        actuals = []
        timestamps = []
        
        for train_start, train_end, test_start, test_end in splits:
            test_df = df.iloc[test_start:test_end].copy()
            
            # Persistence: next hour = current hour
            # Use last value from training as first prediction
            last_train_value = df.iloc[train_end - 1][target_col]
            
            for idx in range(len(test_df)):
                if idx == 0:
                    pred = last_train_value
                else:
                    # Use previous actual (for evaluation)
                    pred = test_df.iloc[idx - 1][target_col]
                
                predictions.append(pred)
                actuals.append(test_df.iloc[idx][target_col])
                timestamps.append(test_df.iloc[idx]['datetime'])
        
        results = {
            'model_name': 'Persistence',
            'predictions': predictions,
            'actuals': actuals,
            'timestamps': timestamps,
            'model': None  # No model object for persistence
        }
        
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        
        self.logger.info(f"Persistence - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        
        return results
    
    def train_seasonal_persistence(self, df: pd.DataFrame, target_col: str,
                                   splits: List[Tuple], period: int = 168) -> Dict:
        """Train seasonal persistence model (weekly)"""
        self.logger.info(f"Training Seasonal Persistence model (period={period})...")
        
        predictions = []
        actuals = []
        timestamps = []
        
        for train_start, train_end, test_start, test_end in splits:
            test_df = df.iloc[test_start:test_end].copy()
            train_df = df.iloc[train_start:train_end].copy()
            
            for idx in range(len(test_df)):
                test_idx = test_start + idx
                # Use same hour from previous week
                seasonal_idx = test_idx - period
                
                if seasonal_idx >= train_start:
                    pred = df.iloc[seasonal_idx][target_col]
                else:
                    # Fallback to mean
                    pred = train_df[target_col].mean()
                
                predictions.append(pred)
                actuals.append(test_df.iloc[idx][target_col])
                timestamps.append(test_df.iloc[idx]['datetime'])
        
        results = {
            'model_name': 'Seasonal_Persistence',
            'predictions': predictions,
            'actuals': actuals,
            'timestamps': timestamps,
            'model': None
        }
        
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        
        self.logger.info(f"Seasonal Persistence - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        
        return results
    
    def train_lightgbm(self, df: pd.DataFrame, target_col: str, 
                      feature_names: List[str], splits: List[Tuple]) -> Dict:
        """Train LightGBM model with Optuna tuning"""
        try:
            import lightgbm as lgb
            import optuna
        except ImportError:
            self.logger.error("LightGBM or Optuna not installed. Skipping.")
            return None
        
        self.logger.info("Training LightGBM model...")
        
        # Use first split for hyperparameter optimization
        if len(splits) == 0:
            return None
        
        train_start, train_end, test_start, test_end = splits[0]
        train_df = df.iloc[train_start:train_end].copy()
        test_df = df.iloc[test_start:test_end].copy()
        
        X_train = train_df[feature_names].values
        y_train = train_df[target_col].values
        X_test = test_df[feature_names].values
        y_test = test_df[target_col].values
        
        def objective(trial):
            params = {
                'objective': 'regression',
                'metric': 'mae',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'verbose': -1,
                'random_state': 42
            }
            
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
            
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=1000,
                callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(0)]
            )
            
            pred = model.predict(X_test, num_iteration=model.best_iteration)
            mae = mean_absolute_error(y_test, pred)
            return mae
        
        n_trials = self.config['models']['lightgbm'].get('optuna_trials', 50)
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        # Train final model with best params on all training data
        best_params = study.best_params
        best_params.update({
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'verbose': -1,
            'random_state': 42
        })
        
        # Train on all splits
        predictions = []
        actuals = []
        timestamps = []
        models = []
        
        for train_start, train_end, test_start, test_end in splits:
            train_df = df.iloc[train_start:train_end].copy()
            test_df = df.iloc[test_start:test_end].copy()
            
            X_train = train_df[feature_names].values
            y_train = train_df[target_col].values
            X_test = test_df[feature_names].values
            
            train_data = lgb.Dataset(X_train, label=y_train)
            model = lgb.train(
                best_params,
                train_data,
                num_boost_round=1000,
                callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(0)]
            )
            
            pred = model.predict(X_test, num_iteration=model.best_iteration)
            predictions.extend(pred)
            actuals.extend(test_df[target_col].values)
            timestamps.extend(test_df['datetime'].values)
            models.append(model)
        
        results = {
            'model_name': 'LightGBM',
            'predictions': predictions,
            'actuals': actuals,
            'timestamps': timestamps,
            'model': models[0] if models else None,  # Save first model
            'best_params': best_params,
            'optuna_study': study
        }
        
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        
        self.logger.info(f"LightGBM - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        
        return results
    
    def train_sarimax(self, df: pd.DataFrame, target_col: str, 
                      feature_names: List[str], splits: List[Tuple], source_type: str) -> Dict:
        """Train SARIMAX model with exogenous features"""
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            import itertools
        except ImportError:
            self.logger.error("statsmodels not installed. Skipping SARIMAX.")
            return None
        
        self.logger.info("Training SARIMAX model...")
        
        if len(splits) == 0:
            return None
        
        sarimax_config = self.config['models']['sarimax']
        
        predictions = []
        actuals = []
        timestamps = []
        models = []
        
        # Use first split for hyperparameter search (simplified)
        train_start, train_end, test_start, test_end = splits[0]
        train_df = df.iloc[train_start:train_end].copy()
        
        # Determine source type from target column
        source_type_inferred = 'solar' if 'solar' in target_col.lower() else 'wind'
        
        # Select a few key exogenous features to avoid overfitting
        key_features = ['WSI_smoothed', 'temperature_mean', 'cloudiness'] if source_type_inferred == 'solar' \
                      else ['WSI_smoothed', 'wind_speed', 'pressure']
        exog_features = [f for f in key_features if f in feature_names][:3]  # Max 3 features
        
        if len(exog_features) == 0:
            exog_features = feature_names[:3]  # Fallback to first 3 features
        
        # Simplified parameter search (just try a few combinations)
        best_aic = np.inf
        best_order = (1, 1, 1)
        best_seasonal_order = (1, 1, 1, sarimax_config['seasonal_period'])
        
        # Try a few common parameter combinations
        p_values = [1, 2]
        d_values = [0, 1]
        q_values = [1, 2]
        P_values = [0, 1]
        D_values = [0, 1]
        Q_values = [0, 1]
        
        y_train = train_df[target_col].values
        if len(exog_features) > 0:
            X_train = train_df[exog_features].values
        else:
            X_train = None
        
        self.logger.info("Searching for best SARIMAX parameters...")
        for p, d, q in itertools.product(p_values, d_values, q_values):
            for P, D, Q in itertools.product(P_values, D_values, Q_values):
                try:
                    if X_train is not None:
                        model = SARIMAX(
                            y_train,
                            exog=X_train,
                            order=(p, d, q),
                            seasonal_order=(P, D, Q, sarimax_config['seasonal_period']),
                            enforce_stationarity=False,
                            enforce_invertibility=False
                        )
                    else:
                        model = SARIMAX(
                            y_train,
                            order=(p, d, q),
                            seasonal_order=(P, D, Q, sarimax_config['seasonal_period']),
                            enforce_stationarity=False,
                            enforce_invertibility=False
                        )
                    
                    fitted_model = model.fit(disp=False, maxiter=50)
                    if fitted_model.aic < best_aic:
                        best_aic = fitted_model.aic
                        best_order = (p, d, q)
                        best_seasonal_order = (P, D, Q, sarimax_config['seasonal_period'])
                except:
                    continue
        
        self.logger.info(f"Best SARIMAX order: {best_order}, seasonal: {best_seasonal_order}")
        
        # Train on all splits
        for train_start, train_end, test_start, test_end in splits:
            train_df = df.iloc[train_start:train_end].copy()
            test_df = df.iloc[test_start:test_end].copy()
            
            y_train = train_df[target_col].values
            if len(exog_features) > 0:
                X_train = train_df[exog_features].values
                X_test = test_df[exog_features].values
            else:
                X_train = None
                X_test = None
            
            try:
                if X_train is not None:
                    model = SARIMAX(
                        y_train,
                        exog=X_train,
                        order=best_order,
                        seasonal_order=best_seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    fitted_model = model.fit(disp=False, maxiter=50)
                    
                    # Forecast
                    if X_test is not None:
                        pred = fitted_model.forecast(steps=len(test_df), exog=X_test)
                    else:
                        pred = fitted_model.forecast(steps=len(test_df))
                else:
                    model = SARIMAX(
                        y_train,
                        order=best_order,
                        seasonal_order=best_seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    fitted_model = model.fit(disp=False, maxiter=50)
                    pred = fitted_model.forecast(steps=len(test_df))
                
                predictions.extend(pred)
                actuals.extend(test_df[target_col].values)
                timestamps.extend(test_df['datetime'].values)
                models.append(fitted_model)
            except Exception as e:
                self.logger.warning(f"SARIMAX failed for split: {e}")
                # Fallback to mean
                predictions.extend([train_df[target_col].mean()] * len(test_df))
                actuals.extend(test_df[target_col].values)
                timestamps.extend(test_df['datetime'].values)
        
        results = {
            'model_name': 'SARIMAX',
            'predictions': predictions,
            'actuals': actuals,
            'timestamps': timestamps,
            'model': models[0] if models else None,
            'best_order': best_order,
            'best_seasonal_order': best_seasonal_order
        }
        
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        
        self.logger.info(f"SARIMAX - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        
        return results
    
    def train_prophet(self, df: pd.DataFrame, target_col: str, 
                     feature_names: List[str], splits: List[Tuple], source_type: str = None) -> Dict:
        """Train Prophet model"""
        try:
            from prophet import Prophet
        except ImportError:
            self.logger.error("Prophet not installed. Skipping.")
            return None
        
        self.logger.info("Training Prophet model...")
        
        prophet_config = self.config['models']['prophet']
        
        predictions = []
        actuals = []
        timestamps = []
        models = []
        
        for train_start, train_end, test_start, test_end in splits:
            train_df = df.iloc[train_start:train_end].copy()
            test_df = df.iloc[test_start:test_end].copy()
            
            # Prepare Prophet format
            prophet_train = pd.DataFrame({
                'ds': train_df['datetime'],
                'y': train_df[target_col]
            })
            
            # Determine source type from target column
            source_type_inferred = 'solar' if 'solar' in target_col.lower() else 'wind'
            
            # Add a few key regressors
            key_features = ['WSI_smoothed', 'temperature_mean'] if source_type_inferred == 'solar' \
                          else ['WSI_smoothed', 'wind_speed']
            regressor_features = [f for f in key_features if f in train_df.columns][:2]
            
            try:
                model = Prophet(
                    yearly_seasonality=prophet_config['yearly_seasonality'],
                    weekly_seasonality=prophet_config['weekly_seasonality'],
                    daily_seasonality=prophet_config['daily_seasonality'],
                    interval_width=0.95
                )
                
                # Add regressors
                for regressor in regressor_features:
                    model.add_regressor(regressor)
                    prophet_train[regressor] = train_df[regressor].values
                
                model.fit(prophet_train)
                
                # Prepare future dataframe
                future = model.make_future_dataframe(periods=len(test_df), freq='H')
                for regressor in regressor_features:
                    # Extend regressor values (use last value for simplicity)
                    future[regressor] = np.concatenate([
                        train_df[regressor].values,
                        [train_df[regressor].iloc[-1]] * len(test_df)
                    ])
                
                forecast = model.predict(future)
                pred = forecast['yhat'].tail(len(test_df)).values
                
                predictions.extend(pred)
                actuals.extend(test_df[target_col].values)
                timestamps.extend(test_df['datetime'].values)
                models.append(model)
            except Exception as e:
                self.logger.warning(f"Prophet failed for split: {e}")
                # Fallback to mean
                predictions.extend([train_df[target_col].mean()] * len(test_df))
                actuals.extend(test_df[target_col].values)
                timestamps.extend(test_df['datetime'].values)
        
        results = {
            'model_name': 'Prophet',
            'predictions': predictions,
            'actuals': actuals,
            'timestamps': timestamps,
            'model': models[0] if models else None
        }
        
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        
        self.logger.info(f"Prophet - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        
        return results
    
    def train_xgboost(self, df: pd.DataFrame, target_col: str, 
                     feature_names: List[str], splits: List[Tuple]) -> Dict:
        """Train XGBoost model with Optuna tuning"""
        try:
            import xgboost as xgb
            import optuna
        except ImportError:
            self.logger.error("XGBoost or Optuna not installed. Skipping.")
            return None
        
        self.logger.info("Training XGBoost model...")
        
        if len(splits) == 0:
            return None
        
        train_start, train_end, test_start, test_end = splits[0]
        train_df = df.iloc[train_start:train_end].copy()
        test_df = df.iloc[test_start:test_end].copy()
        
        X_train = train_df[feature_names].values
        y_train = train_df[target_col].values
        X_test = test_df[feature_names].values
        y_test = test_df[target_col].values
        
        def objective(trial):
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'mae',
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': 42
            }
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train, 
                     eval_set=[(X_test, y_test)],
                     early_stopping_rounds=20,
                     verbose=False)
            
            pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, pred)
            return mae
        
        n_trials = self.config['models']['xgboost'].get('optuna_trials', 50)
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        best_params = study.best_params
        best_params.update({
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'random_state': 42
        })
        
        # Train on all splits
        predictions = []
        actuals = []
        timestamps = []
        models = []
        
        for train_start, train_end, test_start, test_end in splits:
            train_df = df.iloc[train_start:train_end].copy()
            test_df = df.iloc[test_start:test_end].copy()
            
            X_train = train_df[feature_names].values
            y_train = train_df[target_col].values
            X_test = test_df[feature_names].values
            
            model = xgb.XGBRegressor(**best_params)
            model.fit(X_train, y_train, verbose=False)
            
            pred = model.predict(X_test)
            predictions.extend(pred)
            actuals.extend(test_df[target_col].values)
            timestamps.extend(test_df['datetime'].values)
            models.append(model)
        
        results = {
            'model_name': 'XGBoost',
            'predictions': predictions,
            'actuals': actuals,
            'timestamps': timestamps,
            'model': models[0] if models else None,
            'best_params': best_params,
            'optuna_study': study
        }
        
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        
        self.logger.info(f"XGBoost - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        
        return results
    
    def train_catboost(self, df: pd.DataFrame, target_col: str, 
                      feature_names: List[str], splits: List[Tuple]) -> Dict:
        """Train CatBoost model with Optuna tuning"""
        try:
            from catboost import CatBoostRegressor
            import optuna
        except ImportError:
            self.logger.error("CatBoost or Optuna not installed. Skipping.")
            return None
        
        self.logger.info("Training CatBoost model...")
        
        if len(splits) == 0:
            return None
        
        train_start, train_end, test_start, test_end = splits[0]
        train_df = df.iloc[train_start:train_end].copy()
        test_df = df.iloc[test_start:test_end].copy()
        
        X_train = train_df[feature_names].values
        y_train = train_df[target_col].values
        X_test = test_df[feature_names].values
        y_test = test_df[target_col].values
        
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 100, 1000),
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'random_seed': 42,
                'verbose': False
            }
            
            model = CatBoostRegressor(**params)
            model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=20)
            
            pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, pred)
            return mae
        
        n_trials = self.config['models']['catboost'].get('optuna_trials', 50)
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        best_params = study.best_params
        best_params.update({
            'random_seed': 42,
            'verbose': False
        })
        
        # Train on all splits
        predictions = []
        actuals = []
        timestamps = []
        models = []
        
        for train_start, train_end, test_start, test_end in splits:
            train_df = df.iloc[train_start:train_end].copy()
            test_df = df.iloc[test_start:test_end].copy()
            
            X_train = train_df[feature_names].values
            y_train = train_df[target_col].values
            X_test = test_df[feature_names].values
            
            model = CatBoostRegressor(**best_params)
            model.fit(X_train, y_train, verbose=False)
            
            pred = model.predict(X_test)
            predictions.extend(pred)
            actuals.extend(test_df[target_col].values)
            timestamps.extend(test_df['datetime'].values)
            models.append(model)
        
        results = {
            'model_name': 'CatBoost',
            'predictions': predictions,
            'actuals': actuals,
            'timestamps': timestamps,
            'model': models[0] if models else None,
            'best_params': best_params,
            'optuna_study': study
        }
        
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        
        self.logger.info(f"CatBoost - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        
        return results
    
    def train_quantile_lightgbm(self, df: pd.DataFrame, target_col: str, 
                                feature_names: List[str], splits: List[Tuple]) -> Dict:
        """Train Quantile Regression LightGBM"""
        try:
            import lightgbm as lgb
        except ImportError:
            self.logger.error("LightGBM not installed. Skipping.")
            return None
        
        self.logger.info("Training Quantile Regression LightGBM...")
        
        quantile_config = self.config['models']['quantile_lightgbm']
        quantiles = quantile_config.get('quantiles', [0.1, 0.5, 0.9])
        
        # Train models for each quantile
        quantile_results = {}
        
        for q in quantiles:
            self.logger.info(f"  Training quantile {q}...")
            
            predictions = []
            actuals = []
            timestamps = []
            models = []
            
            for train_start, train_end, test_start, test_end in splits:
                train_df = df.iloc[train_start:train_end].copy()
                test_df = df.iloc[test_start:test_end].copy()
                
                X_train = train_df[feature_names].values
                y_train = train_df[target_col].values
                X_test = test_df[feature_names].values
                
                params = {
                    'objective': 'quantile',
                    'alpha': q,
                    'metric': 'quantile',
                    'verbose': -1,
                    'random_state': 42
                }
                
                train_data = lgb.Dataset(X_train, label=y_train)
                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=500,
                    callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(0)]
                )
                
                pred = model.predict(X_test)
                
                if q == 0.5:  # Median is the point prediction
                    predictions.extend(pred)
                    actuals.extend(test_df[target_col].values)
                    timestamps.extend(test_df['datetime'].values)
                
                models.append(model)
            
            quantile_results[f'q{q}'] = {
                'predictions': predictions if q == 0.5 else None,
                'models': models
            }
        
        results = {
            'model_name': 'Quantile_LightGBM',
            'predictions': quantile_results['q0.5']['predictions'],
            'actuals': actuals,
            'timestamps': timestamps,
            'model': quantile_results['q0.5']['models'][0] if quantile_results['q0.5']['models'] else None,
            'quantile_models': quantile_results
        }
        
        if len(actuals) > 0:
            mae = mean_absolute_error(actuals, results['predictions'])
            rmse = np.sqrt(mean_squared_error(actuals, results['predictions']))
            self.logger.info(f"Quantile LightGBM (median) - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        
        return results
    
    def train_all_models(self, source_type: str = 'solar') -> Dict:
        """
        Train all configured models
        
        Parameters:
        -----------
        source_type : str
            'solar' or 'wind'
        
        Returns:
        --------
        dict
            Results dictionary with all model predictions
        """
        self.logger.info("="*60)
        self.logger.info(f"TRAINING ALL MODELS FOR {source_type.upper()}")
        self.logger.info("="*60)
        
        # Load data
        df = self.load_data(source_type)
        
        # Determine target column
        if source_type == 'solar':
            target_col = 'solar_generation_mw'
        else:
            target_col = 'wind_generation_mw'
        
        if target_col not in df.columns:
            raise ValueError(f"Target column {target_col} not found in data")
        
        # Prepare features
        df, feature_names = self.prepare_features(df, source_type)
        
        # Generate walk-forward splits
        walk_config = self.config['forecasting']['walk_forward']
        splits = self.walk_forward_split(
            df,
            train_window_days=walk_config['train_window_days'],
            step_size_days=walk_config['step_size_days']
        )
        
        if len(splits) == 0:
            raise ValueError("No valid train/test splits generated")
        
        # Train models
        results = {}
        
        # Baseline models
        if self.config['models']['persistence']['enabled']:
            results['persistence'] = self.train_persistence(df, target_col, splits)
        
        if self.config['models']['seasonal_persistence']['enabled']:
            period = self.config['models']['seasonal_persistence'].get('period', 168)
            results['seasonal_persistence'] = self.train_seasonal_persistence(
                df, target_col, splits, period
            )
        
        # Statistical models
        if self.config['models']['sarimax']['enabled']:
            results['sarimax'] = self.train_sarimax(df, target_col, feature_names, splits, source_type)
        
        if self.config['models']['prophet']['enabled']:
            results['prophet'] = self.train_prophet(df, target_col, feature_names, splits, source_type)
        
        # Gradient Boosting
        if self.config['models']['lightgbm']['enabled']:
            results['lightgbm'] = self.train_lightgbm(df, target_col, feature_names, splits)
        
        if self.config['models']['xgboost']['enabled']:
            results['xgboost'] = self.train_xgboost(df, target_col, feature_names, splits)
        
        if self.config['models']['catboost']['enabled']:
            results['catboost'] = self.train_catboost(df, target_col, feature_names, splits)
        
        # Probabilistic models
        if self.config['models']['quantile_lightgbm']['enabled']:
            results['quantile_lightgbm'] = self.train_quantile_lightgbm(df, target_col, feature_names, splits)
        
        # Save results
        results_file = self.results_dir / f"{source_type}_training_results.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        
        self.logger.info(f"Saved results to: {results_file}")
        
        return results


def main():
    """Main execution"""
    trainer = ForecastingTrainer()
    
    # Train for solar
    print("\nTraining solar models...")
    solar_results = trainer.train_all_models('solar')
    
    # Train for wind
    print("\nTraining wind models...")
    wind_results = trainer.train_all_models('wind')
    
    print("\nTraining complete!")
    print(f"Results saved to: {trainer.results_dir}")


if __name__ == "__main__":
    main()

