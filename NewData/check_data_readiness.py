import pandas as pd
import numpy as np

print("=" * 60)
print("DATA READINESS CHECK FOR MODEL TRAINING")
print("=" * 60)

# Load training data
print("\n1. Loading training datasets...")
wind_train = pd.read_csv('Trainingdata_5years/Wind_data.csv')
solar_train = pd.read_csv('Trainingdata_5years/Solar_data.csv')

# Load target variables
print("2. Loading target variables...")
wind_prod = pd.read_csv('energy production/merged_Wind.csv')
solar_prod = pd.read_csv('energy production/merged_Solar.csv')

# Convert datetime
wind_train['datetime'] = pd.to_datetime(wind_train['datetime'])
solar_train['datetime'] = pd.to_datetime(solar_train['datetime'])
wind_prod['datetime'] = pd.to_datetime(wind_prod['datetime'])
solar_prod['datetime'] = pd.to_datetime(solar_prod['datetime'])

print("\n" + "=" * 60)
print("WIND DATA ANALYSIS")
print("=" * 60)
print(f"Training data shape: {wind_train.shape}")
print(f"Training data columns: {list(wind_train.columns)}")
print(f"Missing values: {wind_train.isnull().sum().sum()}")
print(f"Date range: {wind_train['datetime'].min()} to {wind_train['datetime'].max()}")
print(f"\nProduction data shape: {wind_prod.shape}")
print(f"Production data columns: {list(wind_prod.columns)}")
print(f"Date range: {wind_prod['datetime'].min()} to {wind_prod['datetime'].max()}")

# Check merge
merged_wind = pd.merge(wind_train, wind_prod, on='datetime', how='inner')
print(f"\nMerged wind data: {merged_wind.shape[0]} rows")
if merged_wind.shape[0] == wind_train.shape[0]:
    print("✓ Perfect alignment!")
else:
    print(f"⚠ Warning: {wind_train.shape[0] - merged_wind.shape[0]} rows missing after merge")

print("\n" + "=" * 60)
print("SOLAR DATA ANALYSIS")
print("=" * 60)
print(f"Training data shape: {solar_train.shape}")
print(f"Training data columns: {list(solar_train.columns)}")
print(f"Missing values: {solar_train.isnull().sum().sum()}")
print(f"Date range: {solar_train['datetime'].min()} to {solar_train['datetime'].max()}")
print(f"\nProduction data shape: {solar_prod.shape}")
print(f"Production data columns: {list(solar_prod.columns)}")
print(f"Date range: {solar_prod['datetime'].min()} to {solar_prod['datetime'].max()}")

# Check merge
merged_solar = pd.merge(solar_train, solar_prod, on='datetime', how='inner')
print(f"\nMerged solar data: {merged_solar.shape[0]} rows")
if merged_solar.shape[0] == solar_train.shape[0]:
    print("✓ Perfect alignment!")
else:
    print(f"⚠ Warning: {solar_train.shape[0] - merged_solar.shape[0]} rows missing after merge")

print("\n" + "=" * 60)
print("READINESS ASSESSMENT")
print("=" * 60)

issues = []
ready = True

# Check 1: Target variables present
if 'solar' not in merged_solar.columns:
    issues.append("❌ Solar target variable 'solar' not found in merged data")
    ready = False
else:
    print("✓ Solar target variable found")

if 'Wind Offshore' not in merged_wind.columns or 'Wind Onshore' not in merged_wind.columns:
    issues.append("❌ Wind target variables not found in merged data")
    ready = False
else:
    print("✓ Wind target variables found")

# Check 2: Missing values
if merged_solar.isnull().sum().sum() > 0:
    issues.append(f"⚠ Solar data has {merged_solar.isnull().sum().sum()} missing values")
    print(f"⚠ Solar data has missing values: {merged_solar.isnull().sum().sum()}")
else:
    print("✓ Solar data has no missing values")

if merged_wind.isnull().sum().sum() > 0:
    issues.append(f"⚠ Wind data has {merged_wind.isnull().sum().sum()} missing values")
    print(f"⚠ Wind data has missing values: {merged_wind.isnull().sum().sum()}")
else:
    print("✓ Wind data has no missing values")

# Check 3: Data size
if merged_solar.shape[0] < 1000:
    issues.append(f"⚠ Solar data has only {merged_solar.shape[0]} rows (may be too small)")
    print(f"⚠ Solar data size: {merged_solar.shape[0]} rows")
else:
    print(f"✓ Solar data size: {merged_solar.shape[0]} rows (sufficient)")

if merged_wind.shape[0] < 1000:
    issues.append(f"⚠ Wind data has only {merged_wind.shape[0]} rows (may be too small)")
    print(f"⚠ Wind data size: {merged_wind.shape[0]} rows")
else:
    print(f"✓ Wind data size: {merged_wind.shape[0]} rows (sufficient)")

# Check 4: Feature engineering
if all(col in merged_solar.columns for col in ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'doy_sin', 'doy_cos']):
    print("✓ Solar data has time cycle features")
else:
    issues.append("❌ Solar data missing time cycle features")
    ready = False

if all(col in merged_wind.columns for col in ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'doy_sin', 'doy_cos']):
    print("✓ Wind data has time cycle features")
else:
    issues.append("❌ Wind data missing time cycle features")
    ready = False

print("\n" + "=" * 60)
if ready and len(issues) == 0:
    print("✅ DATA IS READY FOR MODEL TRAINING!")
    print("\nNext steps:")
    print("1. Create train/validation/test splits")
    print("2. Scale/normalize features if needed")
    print("3. Train models (e.g., Random Forest, XGBoost, LSTM)")
else:
    print("⚠ DATA NEEDS PREPARATION BEFORE TRAINING")
    print("\nIssues to address:")
    for issue in issues:
        print(f"  {issue}")
print("=" * 60)
