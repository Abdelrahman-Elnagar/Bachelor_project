import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import pickle
import warnings
warnings.filterwarnings('ignore')


# Read data
data = pd.read_csv('Data/training_data/model_training_data_with_wsi.csv')
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
print("TRAINING MODELS ON SOLAR GENERATION (WITH WSI)")
print("="*60)

# Prepare data without wind_generation_mw
data_solar = data.drop(columns=['wind_generation_mw'])
target_solar = data_solar['solar_generation_mw']

print("\nChecking stationarity of solar_generation_mw:")
is_stationary_solar = check_stationarity(target_solar)
print(f"Is stationary: {is_stationary_solar}\n")

# Split data (80/20 train/test)
train_size = int(len(target_solar) * 0.8)
train = target_solar[:train_size]
test = target_solar[train_size:]

print(f"Training size: {len(train)}")
print(f"Test size: {len(test)}\n")

# Train ARIMA model
print("Training ARIMA model for solar generation...")
arima_model = ARIMA(train, order=(5, 1, 2))
arima_fit = arima_model.fit()
print("ARIMA model for solar trained successfully")
print()

# Save ARIMA model
with open('forecasting/models/arima_solar_with_wsi_model.pkl', 'wb') as f:
    pickle.dump(arima_fit, f)
print("ARIMA solar (with WSI) model saved\n")

# Train SARIMA model
print("Training SARIMA model for solar generation...")
sarima_model = SARIMAX(train, order=(5, 1, 2), seasonal_order=(1, 1, 1, 24))
sarima_fit = sarima_model.fit(disp=False)
print("SARIMA model for solar trained successfully")
print()

# Save SARIMA model
with open('forecasting/models/sarima_solar_with_wsi_model.pkl', 'wb') as f:
    pickle.dump(sarima_fit, f)
print("SARIMA solar (with WSI) model saved\n")

# Make predictions
arima_pred = arima_fit.forecast(steps=len(test))
sarima_pred = sarima_fit.forecast(steps=len(test))

# Calculate errors
arima_errors = test.values - arima_pred.values
sarima_errors = test.values - sarima_pred.values

# Create results DataFrame
results_df = pd.DataFrame({
    'datetime': test.index,
    'actual': test.values,
    'arima_prediction': arima_pred.values,
    'arima_error': arima_errors,
    'sarima_prediction': sarima_pred.values,
    'sarima_error': sarima_errors
})

# Save to CSV
results_df.to_csv('forecasting/models/results_solar_with_wsi.csv', index=False)
print("Results saved to results_solar_with_wsi.csv\n")

print("SOLAR (WITH WSI) - ARIMA Metrics:")
arima_metrics = calculate_metrics(test, arima_pred)
for metric, value in arima_metrics.items():
    print(f"{metric}: {value:.6f}")
print()

print("SOLAR (WITH WSI) - SARIMA Metrics:")
sarima_metrics = calculate_metrics(test, sarima_pred)
for metric, value in sarima_metrics.items():
    print(f"{metric}: {value:.6f}")
print()

print("="*60)
print("SOLAR (WITH WSI) TRAINING COMPLETE!")
print("="*60)

