import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import pickle
import warnings
warnings.filterwarnings('ignore')


# Read data
data = pd.read_csv('Data/training_data/model_training_data_no_wsi.csv')
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

# Check stationarity
def check_stationarity(timeseries):
    result = adfuller(timeseries.dropna())
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    return result[1] < 0.05

# Calculate metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error

def calculate_metrics(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae}


print("="*60)
print("TRAINING ARIMAX/SARIMAX ON SOLAR GENERATION (NO WSI)")
print("="*60)

# Prepare data - remove wind_generation_mw and solar_generation_mw for features
data_solar = data.drop(columns=['wind_generation_mw'])
target_solar = data_solar['solar_generation_mw']
exog_features = data_solar.drop(columns=['solar_generation_mw'])

print(f"\nTarget: solar_generation_mw")
print(f"Exogenous features: {list(exog_features.columns)}")
print(f"Number of exogenous features: {len(exog_features.columns)}\n")

print("Checking stationarity of solar_generation_mw:")
is_stationary_solar = check_stationarity(target_solar)
print(f"Is stationary: {is_stationary_solar}\n")

# Split data (80/20 train/test)
train_size = int(len(target_solar) * 0.8)
train = target_solar[:train_size]
test = target_solar[train_size:]
train_exog = exog_features[:train_size]
test_exog = exog_features[train_size:]

print(f"Training size: {len(train)}")
print(f"Test size: {len(test)}\n")

# Train ARIMAX model (SARIMAX without seasonal component)
print("Training ARIMAX model for solar generation...")
arimax_model = SARIMAX(train, exog=train_exog, order=(5, 1, 2), seasonal_order=(0, 0, 0, 0))
arimax_fit = arimax_model.fit(disp=False)
print("ARIMAX model for solar trained successfully")
print()

# Save ARIMAX model
with open('forecasting/models/arimax_solar_no_wsi_model.pkl', 'wb') as f:
    pickle.dump(arimax_fit, f)
print("ARIMAX solar (no WSI) model saved\n")

# Train SARIMAX model with exogenous variables
print("Training SARIMAX model for solar generation...")
sarimax_model = SARIMAX(train, exog=train_exog, order=(5, 1, 2), seasonal_order=(1, 1, 1, 24))
sarimax_fit = sarimax_model.fit(disp=False)
print("SARIMAX model for solar trained successfully")
print()

# Save SARIMAX model
with open('forecasting/models/sarimax_solar_no_wsi_model.pkl', 'wb') as f:
    pickle.dump(sarimax_fit, f)
print("SARIMAX solar (no WSI) model saved\n")

# Make predictions
arimax_pred = arimax_fit.forecast(steps=len(test), exog=test_exog)
sarimax_pred = sarimax_fit.forecast(steps=len(test), exog=test_exog)

# Calculate errors
arimax_errors = test.values - arimax_pred.values
sarimax_errors = test.values - sarimax_pred.values

# Create results DataFrame
results_df = pd.DataFrame({
    'datetime': test.index,
    'actual': test.values,
    'arimax_prediction': arimax_pred.values,
    'arimax_error': arimax_errors,
    'arimax_squared_error': arimax_errors ** 2,
    'sarimax_prediction': sarimax_pred.values,
    'sarimax_error': sarimax_errors,
    'sarimax_squared_error': sarimax_errors ** 2
})

# Save to CSV
results_df.to_csv('forecasting/models/results_arimax_solar_no_wsi.csv', index=False)
print("Results saved to results_arimax_solar_no_wsi.csv\n")

print("SOLAR (NO WSI) - ARIMAX Metrics:")
arimax_metrics = calculate_metrics(test, arimax_pred)
for metric, value in arimax_metrics.items():
    print(f"{metric}: {value:.6f}")
print()

print("SOLAR (NO WSI) - SARIMAX Metrics:")
sarimax_metrics = calculate_metrics(test, sarimax_pred)
for metric, value in sarimax_metrics.items():
    print(f"{metric}: {value:.6f}")
print()

print("="*60)
print("SOLAR (NO WSI) ARIMAX/SARIMAX TRAINING COMPLETE!")
print("="*60)



