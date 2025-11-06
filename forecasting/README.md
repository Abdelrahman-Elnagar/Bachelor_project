# Renewable Energy Forecasting Pipeline

This directory contains the complete implementation for solar PV and wind power forecasting integrated with the Weather Stability Index (WSI).

## Directory Structure

```
forecasting/
├── Scripts/              # All Python scripts
│   ├── data_ingest_energy.py
│   ├── train_forecasting.py
│   ├── evaluate_by_stability.py
│   └── visualize_forecasting_results.py
├── data/
│   ├── raw/             # Raw energy data files
│   ├── processed/        # Processed and merged datasets
│   └── config/           # Configuration files
├── models/               # Trained model artifacts
├── results/              # Evaluation results and reports
├── figures/              # Generated visualizations
└── logs/                 # Execution logs
```

## Setup

### 1. Install Dependencies

```bash
pip install pandas numpy scikit-learn lightgbm xgboost catboost optuna
pip install torch torchvision torchaudio  # For neural models
pip install prophet statsmodels  # For baseline models
pip install shap  # For feature attribution
pip install pyyaml requests  # For configuration and data fetching
```

### 2. Configure Data Sources

Edit `data/config/forecasting.yaml`:

- **CSV Files**: Place solar and wind CSV files in `data/raw/` or specify paths in config
- **ENTSO-E API**: Set `data_sources.entsoe.enabled: true` and provide API key

### 3. Prepare WSI Data

Ensure the WSI pipeline has been run and stability labels exist at:
- `../../Data/processed/stability_labels.csv`

## Usage

### Step 1: Ingest Energy Data

```bash
cd forecasting
python Scripts/data_ingest_energy.py
```

This will:
- Load solar and wind generation data
- Validate completeness and quality
- Detect problems (gaps, outliers, misalignments)
- Preprocess (missing values, timezone alignment)
- Merge with WSI stability labels
- Generate validation reports

### Step 2: Train Forecasting Models

```bash
python Scripts/train_forecasting.py
```

This will train all configured models using walk-forward validation and Optuna hyperparameter tuning.

### Step 3: Evaluate by Stability

```bash
python Scripts/evaluate_by_stability.py
```

This will:
- Compute metrics stratified by stability regime
- Run statistical tests (Mann-Whitney U, Welch t-test)
- Calculate effect sizes and robustness metrics
- Generate SHAP/attention analyses

### Step 4: Generate Visualizations

```bash
python Scripts/visualize_forecasting_results.py
```

This will create publication-ready figures and copy them to the thesis figures directory.

## Model Architecture

### Baseline Models
- **Persistence**: Next hour = current hour
- **Seasonal Persistence**: Same hour from previous week
- **SARIMAX**: ARIMA with exogenous weather features
- **Prophet**: Facebook's additive seasonality model

### Gradient Boosting (CPU-Optimized)
- **LightGBM**: Fastest, state-of-the-art for tabular time series
- **XGBoost**: Well-tested alternative
- **CatBoost**: Handles categorical features

### Neural Forecasting (CPU-Optimized)
- **N-BEATS**: Interpretable neural basis expansion
- **TCN**: Temporal Convolutional Network
- **TFT**: Temporal Fusion Transformer with attention

### Probabilistic
- **Quantile Regression LightGBM**: Prediction intervals
- **Conformal Prediction**: Post-hoc calibration

All models are optimized for CPU execution - no GPU required.

## Configuration

Key parameters in `data/config/forecasting.yaml`:

- **Forecasting horizons**: `[1, 3, 6, 24]` hours
- **Training window**: 90 days (configurable)
- **Optuna trials**: Reduced for CPU efficiency (30-50 per model)
- **Model architectures**: Lightweight (hidden dims ≤ 128, fewer layers)

## Output

- **Models**: Saved in `models/` directory
- **Results**: Evaluation metrics, statistical tests in `results/`
- **Figures**: Visualizations in `figures/` and copied to `../../figures/thesis/`
- **Logs**: Execution logs in `logs/`

## Integration with WSI

The forecasting pipeline uses WSI-derived features:
- `WSI_smoothed`: Smoothed stability index
- Rolling WSI statistics (6h, 24h means and stds)
- Stability labels (GMM, K-Means, percentile-based)
- Regime duration and transition features

Evaluation is stratified by stability regime to assess model robustness.

