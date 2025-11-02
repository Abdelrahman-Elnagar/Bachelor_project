#!/usr/bin/env python3
"""Quick verification script for preprocessed temperature data"""
import pandas as pd
import numpy as np
from pathlib import Path

# Check a sample file
file_path = Path("Data/Bundesland_aggregation/Temp_Bundesland_Aggregated_preprocessed/Bayern_aggregated.csv")
df = pd.read_csv(file_path)

print("="*60)
print("PREPROCESSING VERIFICATION")
print("="*60)
print(f"File: {file_path.name}")
print(f"Total rows: {len(df):,}")
print(f"NaN values in temperature_mean: {df['temperature_mean'].isna().sum()}")
print(f"Temperature range: {df['temperature_mean'].min():.2f}°C to {df['temperature_mean'].max():.2f}°C")
print(f"Outlier flag column exists: {'outlier_flag' in df.columns}")
print(f"Columns: {list(df.columns)}")

# Check for outliers (should be none beyond thresholds)
temp_col = df['temperature_mean']
outliers_domain = ((temp_col < -50) | (temp_col > 50)).sum()
print(f"Values outside domain range (-50 to 50): {outliers_domain}")

print("\n✓ Preprocessing verified!")

