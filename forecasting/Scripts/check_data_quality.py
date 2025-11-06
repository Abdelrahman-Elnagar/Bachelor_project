#!/usr/bin/env python3
"""Check data quality for missing values and outliers"""

import pandas as pd
import numpy as np
from pathlib import Path

base_dir = Path(__file__).parent.parent
data_file = base_dir / "data" / "processed" / "energy_production_2024.csv"

df = pd.read_csv(data_file)

print("="*60)
print("DATA QUALITY CHECK")
print("="*60)

print("\nMissing values:")
solar_missing = df['solar_generation_mw'].isna().sum()
wind_missing = df['wind_generation_mw'].isna().sum()
print(f"  Solar: {solar_missing} ({solar_missing/len(df)*100:.3f}%)")
print(f"  Wind: {wind_missing} ({wind_missing/len(df)*100:.3f}%)")

print("\nQuality flags:")
print(f"  Solar missing flag: {df['solar_missing_flag'].sum()}")
print(f"  Solar interpolated flag: {df['solar_interpolated_flag'].sum()}")
print(f"  Wind missing flag: {df['wind_missing_flag'].sum()}")
print(f"  Wind interpolated flag: {df['wind_interpolated_flag'].sum()}")

print("\nOutlier check (Z-score > 4.0):")
solar_z = np.abs((df['solar_generation_mw'] - df['solar_generation_mw'].mean()) / df['solar_generation_mw'].std())
wind_z = np.abs((df['wind_generation_mw'] - df['wind_generation_mw'].mean()) / df['wind_generation_mw'].std())
solar_outliers = (solar_z > 4).sum()
wind_outliers = (wind_z > 4).sum()
print(f"  Solar outliers: {solar_outliers}")
print(f"  Wind outliers: {wind_outliers}")

if solar_outliers > 0:
    print(f"\n  Solar outlier examples (Z > 4):")
    outlier_df = df[solar_z > 4][['datetime', 'solar_generation_mw']].head(5)
    for idx, row in outlier_df.iterrows():
        z_val = solar_z.iloc[idx]
        print(f"    {row['datetime']}: {row['solar_generation_mw']:.1f} MW (Z={z_val:.2f})")

if wind_outliers > 0:
    print(f"\n  Wind outlier examples (Z > 4):")
    outlier_df = df[wind_z > 4][['datetime', 'wind_generation_mw']].head(5)
    for idx, row in outlier_df.iterrows():
        z_val = wind_z.iloc[idx]
        print(f"    {row['datetime']}: {row['wind_generation_mw']:.1f} MW (Z={z_val:.2f})")

print("\nData statistics:")
print(f"  Solar: min={df['solar_generation_mw'].min():.1f}, max={df['solar_generation_mw'].max():.1f}, mean={df['solar_generation_mw'].mean():.1f}")
print(f"  Wind: min={df['wind_generation_mw'].min():.1f}, max={df['wind_generation_mw'].max():.1f}, mean={df['wind_generation_mw'].mean():.1f}")

print("\n" + "="*60)

