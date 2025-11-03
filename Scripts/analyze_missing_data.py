#!/usr/bin/env python3
"""Analyze how missing data was handled in preprocessing"""
import pandas as pd
import numpy as np
from pathlib import Path

# Compare original vs preprocessed
original_file = Path("Data/Bundesland_aggregation/Temp_Bundesland_Aggregated/Bayern_aggregated.csv")
preprocessed_file = Path("Data/Bundesland_aggregation/Temp_Bundesland_Aggregated_preprocessed/Bayern_aggregated.csv")

df_orig = pd.read_csv(original_file)
df_prep = pd.read_csv(preprocessed_file)

print("="*60)
print("MISSING DATA ANALYSIS")
print("="*60)
print("\nORIGINAL DATA:")
print(f"  Total rows: {len(df_orig):,}")

# Check for missing value markers
missing_markers = [-999, -999.0, "-999", "-999.0"]
for marker in missing_markers:
    count = (df_orig['temperature_mean'] == marker).sum()
    if count > 0:
        print(f"  Rows with {marker}: {count}")

# Check for existing NaN
nan_count = df_orig['temperature_mean'].isna().sum()
print(f"  NaN values: {nan_count}")

# Check for invalid values
invalid_count = ((df_orig['temperature_mean'] < -50) | (df_orig['temperature_mean'] > 50)).sum()
print(f"  Values outside domain range (-50 to 50): {invalid_count}")

print("\nPREPROCESSED DATA:")
print(f"  Total rows: {len(df_prep):,}")
print(f"  NaN values: {df_prep['temperature_mean'].isna().sum()}")

# Calculate what was removed
rows_removed = len(df_orig) - len(df_prep)
print(f"\nREMOVED:")
print(f"  Total rows removed: {rows_removed} ({rows_removed/len(df_orig)*100:.2f}%)")

# What happened:
print("\nPROCESSING STEPS:")
print("  1. Missing value markers (-999, -999.0) were replaced with NaN")
print("  2. Outliers were detected and removed")
print("  3. Rows with NaN in temperature_mean were REMOVED entirely")
print("  4. Only rows with valid temperature values remain")

