"""
Merge wind target variables (Wind Offshore, Wind Onshore) into Wind_data.csv
"""

import pandas as pd

print("=" * 60)
print("MERGING WIND TARGET VARIABLES")
print("=" * 60)

# Load the datasets
print("\n1. Loading datasets...")
wind_features = pd.read_csv('Trainingdata_5years/Wind_data.csv')
wind_targets = pd.read_csv('energy production/merged_Wind.csv')

print(f"   Wind features shape: {wind_features.shape}")
print(f"   Wind targets shape: {wind_targets.shape}")

# Convert datetime columns to ensure proper matching
print("\n2. Converting datetime columns...")
wind_features['datetime'] = pd.to_datetime(wind_features['datetime'])
wind_targets['datetime'] = pd.to_datetime(wind_targets['datetime'])

print(f"   Wind features date range: {wind_features['datetime'].min()} to {wind_features['datetime'].max()}")
print(f"   Wind targets date range: {wind_targets['datetime'].min()} to {wind_targets['datetime'].max()}")

# Check for existing target columns
if 'Wind Offshore' in wind_features.columns or 'Wind Onshore' in wind_features.columns:
    print("\n   ⚠ Warning: Target columns already exist in Wind_data.csv")
    print("   They will be overwritten with values from merged_Wind.csv")

# Merge on datetime
print("\n3. Merging on datetime...")
wind_merged = pd.merge(
    wind_features,
    wind_targets[['datetime', 'Wind Offshore', 'Wind Onshore']],
    on='datetime',
    how='inner'  # Use inner join to keep only matching timestamps
)

print(f"   Merged shape: {wind_merged.shape}")
print(f"   Original features rows: {wind_features.shape[0]}")
print(f"   Merged rows: {wind_merged.shape[0]}")

# Check alignment
if wind_merged.shape[0] == wind_features.shape[0]:
    print("   ✓ Perfect alignment - all rows matched!")
else:
    missing = wind_features.shape[0] - wind_merged.shape[0]
    print(f"   ⚠ Warning: {missing} rows from features did not match with targets")

# Verify target columns are present
if 'Wind Offshore' in wind_merged.columns and 'Wind Onshore' in wind_merged.columns:
    print("\n4. Verifying target columns...")
    print(f"   ✓ Wind Offshore: {wind_merged['Wind Offshore'].isnull().sum()} missing values")
    print(f"   ✓ Wind Onshore: {wind_merged['Wind Onshore'].isnull().sum()} missing values")
    
    # Show some statistics
    print(f"\n   Target statistics:")
    print(f"   Wind Offshore - Min: {wind_merged['Wind Offshore'].min():.2f}, Max: {wind_merged['Wind Offshore'].max():.2f}, Mean: {wind_merged['Wind Offshore'].mean():.2f}")
    print(f"   Wind Onshore - Min: {wind_merged['Wind Onshore'].min():.2f}, Max: {wind_merged['Wind Onshore'].max():.2f}, Mean: {wind_merged['Wind Onshore'].mean():.2f}")
else:
    print("   ❌ Error: Target columns not found after merge!")
    exit(1)

# Reorder columns: datetime first, then time cycles, then features, then targets
print("\n5. Reordering columns...")
feature_cols = [col for col in wind_merged.columns 
                if col not in ['datetime', 'Wind Offshore', 'Wind Onshore']]
column_order = ['datetime'] + feature_cols + ['Wind Offshore', 'Wind Onshore']
wind_merged = wind_merged[column_order]

print(f"   Final column order: {list(wind_merged.columns[:5])} ... {list(wind_merged.columns[-3:])}")

# Save the merged dataset
print("\n6. Saving merged dataset...")
output_file = 'Trainingdata_5years/Wind_data.csv'
wind_merged.to_csv(output_file, index=False)
print(f"   ✓ Saved: {output_file}")
print(f"   Final shape: {wind_merged.shape[0]} rows, {wind_merged.shape[1]} columns")

print("\n" + "=" * 60)
print("✅ MERGE COMPLETED SUCCESSFULLY!")
print("=" * 60)
print(f"\nWind_data.csv is now ready for training with:")
print(f"  - {wind_merged.shape[1] - 3} feature columns")
print(f"  - 2 target columns: Wind Offshore, Wind Onshore")
print(f"  - {wind_merged.shape[0]} training samples")
print("=" * 60)
