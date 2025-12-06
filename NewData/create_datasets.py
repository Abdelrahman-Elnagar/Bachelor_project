import pandas as pd
import numpy as np

input_file = r'd:\NewData\merged_all_attributes_no_soil.csv'

# Define base columns for each dataset
solar_base_columns = [
    'datetime',
    'global_radiation',
    'diffuse_radiation',
    'zenith_angle',
    'sunshine_duration',
    'atmospheric_radiation',
    'avg_temperature',
    'wind_speed',
    'cloudiness',
    'avg_humidity',
    'visibility'
]

wind_base_columns = [
    'datetime',
    'wind_speed',
    'wind_direction',
    'extreme_wind_speed',
    'avg_temperature',
    'pressure_station',
    'avg_humidity'
]

wsi_columns = [
    'datetime',
    'pressure_station',
    'dew_point',
    'avg_temperature',
    'precipitation',
    'extreme_wind_speed',
    'cloudiness',
    'visibility'
]

def add_time_cycles(df):
    """
    Add cyclic time features: hour, month, and day of year cycles.
    """
    # Convert datetime to datetime type if not already
    if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
        df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Extract hour and month
    df['hour'] = df['datetime'].dt.hour
    df['month'] = df['datetime'].dt.month
    
    # 1. Hour of Day Cycle (0-23)
    # 24 hours in a cycle
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # 2. Month of Year Cycle (1-12)
    # 12 months in a cycle
    df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)
    
    # 3. Day of Year Cycle (Seasonal Accuracy)
    # 365.25 days in a cycle. This is often BETTER than 'Month' for solar/wind 
    # because it captures gradual changes (Jan 1 vs Jan 31).
    day_of_year = df['datetime'].dt.dayofyear
    df['doy_sin'] = np.sin(2 * np.pi * day_of_year / 365.25)
    df['doy_cos'] = np.cos(2 * np.pi * day_of_year / 365.25)
    
    # Drop temporary columns
    df = df.drop(columns=['hour', 'month'])
    
    return df

# Read the input file
print(f'Reading input file: {input_file}')
df = pd.read_csv(input_file)

# Verify all required columns exist
all_required = set(solar_base_columns + wind_base_columns + wsi_columns)
missing_columns = all_required - set(df.columns)

if missing_columns:
    print(f'ERROR: Missing columns: {missing_columns}')
    exit(1)

print(f'Processed {len(df)} rows')

# Create Solar dataset
print(f'\nCreating Solar dataset...')
solar_df = df[solar_base_columns].copy()
solar_df = add_time_cycles(solar_df)

# Reorder columns: datetime first, then time cycles, then other features
solar_columns = ['datetime'] + \
                ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'doy_sin', 'doy_cos'] + \
                [col for col in solar_base_columns if col != 'datetime']
solar_df = solar_df[solar_columns]

# Create Wind dataset
print(f'Creating Wind dataset...')
wind_df = df[wind_base_columns].copy()
wind_df = add_time_cycles(wind_df)

# Reorder columns: datetime first, then time cycles, then other features
wind_columns = ['datetime'] + \
               ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'doy_sin', 'doy_cos'] + \
               [col for col in wind_base_columns if col != 'datetime']
wind_df = wind_df[wind_columns]

# Create WSI dataset (no time cycles)
print(f'Creating WSI dataset...')
wsi_df = df[wsi_columns].copy()

# Write Solar_data.csv
solar_output = r'd:\NewData\Solar_data.csv'
print(f'\nWriting {solar_output}...')
solar_df.to_csv(solar_output, index=False)
print(f'  Created with {len(solar_df.columns)} columns and {len(solar_df)} rows')

# Write Wind_data.csv
wind_output = r'd:\NewData\Wind_data.csv'
print(f'\nWriting {wind_output}...')
wind_df.to_csv(wind_output, index=False)
print(f'  Created with {len(wind_df.columns)} columns and {len(wind_df)} rows')

# Write WSI_data.csv
wsi_output = r'd:\NewData\WSI_data.csv'
print(f'\nWriting {wsi_output}...')
wsi_df.to_csv(wsi_output, index=False)
print(f'  Created with {len(wsi_df.columns)} columns and {len(wsi_df)} rows')

print('\n✓ All datasets created successfully!')
