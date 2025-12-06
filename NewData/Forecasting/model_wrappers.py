"""
Model wrapper classes for energy prediction models.
Provides standardized interface for statistical and ML models.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from prophet import Prophet
from tbats import TBATS
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')


class BaseModelWrapper:
    """Base class for all model wrappers."""
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.scaler = None
        
    def fit(self, X_train, y_train, X_test=None, **kwargs):
        """Fit the model. To be implemented by subclasses."""
        raise NotImplementedError
        
    def predict(self, X):
        """Make predictions. To be implemented by subclasses."""
        raise NotImplementedError
        
    def calculate_rmse_per_timestamp(self, y_true, y_pred):
        """Calculate RMSE for each timestamp."""
        return np.sqrt(np.square(y_true - y_pred))


class ProphetWrapper(BaseModelWrapper):
    """Wrapper for Facebook Prophet model."""
    
    def __init__(self):
        super().__init__('Prophet')
        
    def fit(self, X_train, y_train, X_test=None, **kwargs):
        """Fit Prophet model with exogenous regressors."""
        # Prepare data for Prophet
        df = pd.DataFrame({
            'ds': X_train['datetime'],
            'y': y_train.values
        })
        
        # Add exogenous regressors (all features except datetime)
        regressor_cols = [col for col in X_train.columns if col != 'datetime']
        for col in regressor_cols:
            df[col] = X_train[col].values
            
        # Create Prophet model with parameters
        self.model = Prophet(**kwargs)
        
        # Add all regressors
        for col in regressor_cols:
            self.model.add_regressor(col)
            
        self.model.fit(df)
        return self
        
    def predict(self, X):
        """Make predictions with Prophet."""
        # Prepare future dataframe
        future = pd.DataFrame({'ds': X['datetime']})
        
        # Add exogenous regressors
        regressor_cols = [col for col in X.columns if col != 'datetime']
        for col in regressor_cols:
            future[col] = X[col].values
            
        forecast = self.model.predict(future)
        return forecast['yhat'].values


class TBATSWrapper(BaseModelWrapper):
    """Wrapper for TBATS model."""
    
    def __init__(self):
        super().__init__('TBATS')
        self.exog_train = None
        self.exog_test = None
        
    def fit(self, X_train, y_train, X_test=None, **kwargs):
        """Fit TBATS model."""
        # TBATS expects time series data
        y_ts = y_train.values
        
        # Extract exogenous variables (all features except datetime)
        exog_cols = [col for col in X_train.columns if col != 'datetime']
        if exog_cols:
            self.exog_train = X_train[exog_cols].values
            if X_test is not None:
                self.exog_test = X_test[exog_cols].values
        else:
            self.exog_train = None
            self.exog_test = None
            
        # Create and fit TBATS model
        self.model = TBATS(**kwargs)
        self.model = self.model.fit(y_ts, use_exogenous=len(exog_cols) > 0)
        return self
        
    def predict(self, X):
        """Make predictions with TBATS."""
        # Extract exogenous variables if available
        exog_cols = [col for col in X.columns if col != 'datetime']
        if exog_cols and self.exog_train is not None:
            exog = X[exog_cols].values
        else:
            exog = None
            
        forecast = self.model.forecast(steps=len(X), exog=exog)
        return forecast


class SARIMAXWrapper(BaseModelWrapper):
    """Wrapper for SARIMAX model."""
    
    def __init__(self):
        super().__init__('SARIMAX')
        self.exog_train = None
        self.exog_test = None
        
    def fit(self, X_train, y_train, X_test=None, **kwargs):
        """Fit SARIMAX model."""
        # Extract exogenous variables
        exog_cols = [col for col in X_train.columns if col != 'datetime']
        if exog_cols:
            self.exog_train = X_train[exog_cols].values
            if X_test is not None:
                self.exog_test = X_test[exog_cols].values
        else:
            self.exog_train = None
            self.exog_test = None
            
        # Extract order and seasonal_order from kwargs
        order = kwargs.pop('order', (1, 1, 1))
        seasonal_order = kwargs.pop('seasonal_order', (1, 1, 1, 24))
        
        # Create and fit SARIMAX model
        self.model = SARIMAX(
            y_train.values,
            exog=self.exog_train,
            order=order,
            seasonal_order=seasonal_order,
            **kwargs
        )
        self.model = self.model.fit(disp=False)
        return self
        
    def predict(self, X):
        """Make predictions with SARIMAX."""
        # Extract exogenous variables if available
        exog_cols = [col for col in X.columns if col != 'datetime']
        if exog_cols and self.exog_train is not None:
            exog = X[exog_cols].values
        else:
            exog = None
            
        forecast = self.model.forecast(steps=len(X), exog=exog)
        return forecast


class XGBoostWrapper(BaseModelWrapper):
    """Wrapper for XGBoost model."""
    
    def __init__(self):
        super().__init__('XGBoost')
        self.scaler = StandardScaler()
        
    def fit(self, X_train, y_train, X_test=None, **kwargs):
        """Fit XGBoost model."""
        # Extract features (exclude datetime)
        feature_cols = [col for col in X_train.columns if col != 'datetime']
        X_train_features = X_train[feature_cols].values
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_features)
        
        # Create and fit XGBoost model
        self.model = xgb.XGBRegressor(**kwargs, random_state=42, n_jobs=-1)
        self.model.fit(X_train_scaled, y_train.values)
        return self
        
    def predict(self, X):
        """Make predictions with XGBoost."""
        # Extract features
        feature_cols = [col for col in X.columns if col != 'datetime']
        X_features = X[feature_cols].values
        
        # Scale features
        X_scaled = self.scaler.transform(X_features)
        
        return self.model.predict(X_scaled)


class LightGBMWrapper(BaseModelWrapper):
    """Wrapper for LightGBM model."""
    
    def __init__(self):
        super().__init__('LightGBM')
        self.scaler = StandardScaler()
        
    def fit(self, X_train, y_train, X_test=None, **kwargs):
        """Fit LightGBM model."""
        # Extract features
        feature_cols = [col for col in X_train.columns if col != 'datetime']
        X_train_features = X_train[feature_cols].values
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_features)
        
        # Create and fit LightGBM model
        self.model = lgb.LGBMRegressor(**kwargs, random_state=42, n_jobs=-1, verbose=-1)
        self.model.fit(X_train_scaled, y_train.values)
        return self
        
    def predict(self, X):
        """Make predictions with LightGBM."""
        # Extract features
        feature_cols = [col for col in X.columns if col != 'datetime']
        X_features = X[feature_cols].values
        
        # Scale features
        X_scaled = self.scaler.transform(X_features)
        
        return self.model.predict(X_scaled)


class CatBoostWrapper(BaseModelWrapper):
    """Wrapper for CatBoost model."""
    
    def __init__(self):
        super().__init__('CatBoost')
        self.scaler = StandardScaler()
        
    def fit(self, X_train, y_train, X_test=None, **kwargs):
        """Fit CatBoost model."""
        # Extract features
        feature_cols = [col for col in X_train.columns if col != 'datetime']
        X_train_features = X_train[feature_cols].values
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_features)
        
        # Create and fit CatBoost model
        self.model = cb.CatBoostRegressor(**kwargs, random_state=42, verbose=False)
        self.model.fit(X_train_scaled, y_train.values)
        return self
        
    def predict(self, X):
        """Make predictions with CatBoost."""
        # Extract features
        feature_cols = [col for col in X.columns if col != 'datetime']
        X_features = X[feature_cols].values
        
        # Scale features
        X_scaled = self.scaler.transform(X_features)
        
        return self.model.predict(X_scaled)


class RandomForestWrapper(BaseModelWrapper):
    """Wrapper for Random Forest model."""
    
    def __init__(self):
        super().__init__('RandomForest')
        self.scaler = StandardScaler()
        
    def fit(self, X_train, y_train, X_test=None, **kwargs):
        """Fit Random Forest model."""
        # Extract features
        feature_cols = [col for col in X_train.columns if col != 'datetime']
        X_train_features = X_train[feature_cols].values
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_features)
        
        # Create and fit Random Forest model
        self.model = RandomForestRegressor(**kwargs, random_state=42, n_jobs=-1)
        self.model.fit(X_train_scaled, y_train.values)
        return self
        
    def predict(self, X):
        """Make predictions with Random Forest."""
        # Extract features
        feature_cols = [col for col in X.columns if col != 'datetime']
        X_features = X[feature_cols].values
        
        # Scale features
        X_scaled = self.scaler.transform(X_features)
        
        return self.model.predict(X_scaled)
