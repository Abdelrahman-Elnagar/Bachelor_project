# Models and Dependencies Guide

## Overview

All models in this forecasting pipeline are **available through Python libraries** - no manual downloads, pre-trained weights, or API calls required. Everything is installed via `pip install`.

## Model Categories

### 1. Baseline Models (No Installation Needed)
These are simple statistical methods implemented in the code:
- **Persistence**: Next hour = current hour (no library)
- **Seasonal Persistence**: Same hour from previous week (no library)

### 2. Statistical Models (Python Libraries)

#### SARIMAX
- **Library**: `statsmodels`
- **Installation**: `pip install statsmodels`
- **What it does**: Seasonal ARIMA with exogenous variables
- **No pre-trained weights needed**: Model is trained from scratch on your data
- **Status**: ✅ Already in requirements.txt

#### Prophet
- **Library**: `prophet` (Facebook's Prophet)
- **Installation**: `pip install prophet`
- **What it does**: Additive seasonality model for time series
- **No pre-trained weights needed**: Model is trained from scratch on your data
- **Status**: ✅ Already in requirements.txt
- **Note**: May require additional dependencies (pystan, cmdstanpy) on some systems

### 3. Gradient Boosting Models (Python Libraries)

#### LightGBM
- **Library**: `lightgbm`
- **Installation**: `pip install lightgbm`
- **What it does**: Fast gradient boosting framework
- **No pre-trained weights needed**: Model is trained from scratch on your data
- **Status**: ✅ Already in requirements.txt

#### XGBoost
- **Library**: `xgboost`
- **Installation**: `pip install xgboost`
- **What it does**: Extreme gradient boosting
- **No pre-trained weights needed**: Model is trained from scratch on your data
- **Status**: ✅ Already in requirements.txt

#### CatBoost
- **Library**: `catboost`
- **Installation**: `pip install catboost`
- **What it does**: Gradient boosting with categorical feature support
- **No pre-trained weights needed**: Model is trained from scratch on your data
- **Status**: ✅ Already in requirements.txt

### 4. Neural Network Models (Optional - Not Yet Implemented)

#### N-BEATS, TCN, TFT
- **Library**: `torch` (PyTorch)
- **Installation**: `pip install torch torchvision`
- **What they do**: Deep learning models for time series
- **No pre-trained weights needed**: Models are trained from scratch
- **Status**: ⚠️ Framework ready but models not yet implemented
- **CPU-optimized**: All neural models designed to run on CPU (no GPU required)

### 5. Hyperparameter Tuning

#### Optuna
- **Library**: `optuna`
- **Installation**: `pip install optuna`
- **What it does**: Automated hyperparameter optimization
- **Status**: ✅ Already in requirements.txt

### 6. Feature Attribution

#### SHAP
- **Library**: `shap`
- **Installation**: `pip install shap`
- **What it does**: Explains model predictions (feature importance)
- **Status**: ✅ Already in requirements.txt

## Installation

### Quick Install (All Models)
```bash
cd forecasting
pip install -r requirements.txt
```

### Individual Installation
If you want to install models selectively:

```bash
# Statistical models
pip install statsmodels prophet

# Gradient boosting models
pip install lightgbm xgboost catboost

# Hyperparameter tuning
pip install optuna

# Neural networks (optional, for future implementation)
pip install torch

# Feature attribution
pip install shap
```

## What You DON'T Need

❌ **No pre-trained weights or model files to download**
- All models train from scratch on your data
- No need to download .h5, .pkl, .onnx, or other model files

❌ **No API keys or external services**
- Everything runs locally
- No calls to cloud APIs or external services

❌ **No GPU required**
- All models are CPU-optimized
- Neural models will use smaller architectures suitable for CPU

❌ **No manual model downloads**
- Everything is installed via pip
- No need to download models from GitHub or model repositories

## Model Training Process

All models follow this process:

1. **Load your data** (energy production + weather features)
2. **Train from scratch** on your 2024 data
3. **Save trained models** to `forecasting/models/` directory
4. **Use for predictions** on test data

## Current Implementation Status

### ✅ Fully Implemented
- Persistence
- Seasonal Persistence
- SARIMAX
- Prophet
- LightGBM
- XGBoost
- CatBoost
- Quantile Regression LightGBM

### ⚠️ Framework Ready (Not Yet Implemented)
- N-BEATS (neural)
- TCN (neural)
- TFT (neural)

## Dependencies Summary

| Model | Library | Install Command | Status |
|-------|---------|----------------|--------|
| SARIMAX | statsmodels | `pip install statsmodels` | ✅ Ready |
| Prophet | prophet | `pip install prophet` | ✅ Ready |
| LightGBM | lightgbm | `pip install lightgbm` | ✅ Ready |
| XGBoost | xgboost | `pip install xgboost` | ✅ Ready |
| CatBoost | catboost | `pip install catboost` | ✅ Ready |
| Optuna | optuna | `pip install optuna` | ✅ Ready |
| PyTorch | torch | `pip install torch` | ⚠️ For neural models |
| SHAP | shap | `pip install shap` | ✅ Ready |

## Troubleshooting

### Prophet Installation Issues
If `prophet` installation fails, try:
```bash
# Windows
pip install prophet --no-build-isolation

# Or install dependencies separately
pip install pystan cmdstanpy
pip install prophet
```

### LightGBM/XGBoost Installation Issues
If compilation fails, install pre-built wheels:
```bash
pip install --upgrade pip
pip install lightgbm xgboost --only-binary :all:
```

### PyTorch Installation
For CPU-only (recommended):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Verification

After installation, verify all models are available:

```python
# Test imports
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    print("✅ SARIMAX available")
except ImportError:
    print("❌ SARIMAX not installed")

try:
    from prophet import Prophet
    print("✅ Prophet available")
except ImportError:
    print("❌ Prophet not installed")

try:
    import lightgbm as lgb
    print("✅ LightGBM available")
except ImportError:
    print("❌ LightGBM not installed")

try:
    import xgboost as xgb
    print("✅ XGBoost available")
except ImportError:
    print("❌ XGBoost not installed")

try:
    from catboost import CatBoostRegressor
    print("✅ CatBoost available")
except ImportError:
    print("❌ CatBoost not installed")

try:
    import optuna
    print("✅ Optuna available")
except ImportError:
    print("❌ Optuna not installed")
```

## Summary

**All models are available through standard Python libraries.** Just install the requirements and you're ready to train models. No manual downloads, API keys, or pre-trained weights needed!

