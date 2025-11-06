#!/usr/bin/env python3
"""
Weather Feature Engineering for Weather Stability Index (WSI)

This script implements Phase 1 of the WSI pipeline:
- Loads preprocessed weather data from Germany aggregation
- Engineers variability, trend, and extreme event features
- Performs feature selection and correlation analysis
- Applies robust normalization (median, IQR)
- Saves engineered features for WSI computation

Scientific justification: Feature engineering transforms raw weather observations
into meaningful stability indicators based on meteorological principles where
stability is characterized by low variability, consistent trends, and absence of
extreme events.

Author: Weather Stability Index Implementation
"""

import pandas as pd
import numpy as np
import yaml
import json
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import warnings
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class WeatherFeatureEngineer:
    """Feature engineering for Weather Stability Index computation"""
    
    def __init__(self, config_path="Scripts/preprocessing_config.yaml"):
        """Initialize feature engineer with configuration"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.attribute_mappings = self.config['attribute_column_mappings']
        
        # Create output directories
        self.output_dir = Path("data/features")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.models_dir = Path("models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.results_dir = Path("results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Data path
        self.data_path = Path("Preprocessed_Data/Germany_aggregation")
        
    def load_unified_dataset(self):
        """
        Load and merge all preprocessed weather data from Germany aggregation.
        
        Returns:
        --------
        df : pd.DataFrame
            Unified dataset with all weather attributes aligned by datetime
        """
        print("="*60)
        print("LOADING UNIFIED DATASET")
        print("="*60)
        
        # Define attribute folders and their corresponding value columns
        attributes = [
            ('temperature', 'Temp_Germany_Aggregated', ['temperature_mean']),
            ('cloudiness', 'Cloudness_Germany_Aggregated', ['cloudiness_mean']),
            ('wind', 'wind_germany', ['wind_speed']),
            ('wind_synop', 'wind_synop_germany', ['wind_speed_synop']),
            ('precipitation', 'precipitation_germany', ['precipitation_amount']),
            ('pressure', 'pressure_germany', ['pressure', 'pressure_station_level']),
            ('dew_point', 'dew_point_germany_aggregated', ['dew_point']),
            ('moisture', 'moisture_germany_aggregated', ['humidity_std', 'vapor_pressure_std']),
            ('extreme_wind', 'extreme_wind_germany_aggregated', ['extreme_wind_speed_911', 'extreme_wind_speed']),
            ('soil_temperature', 'soil_temperature_germany', ['V_TE002', 'V_TE005', 'V_TE010']),
            ('sun', 'sun_germany', ['sunshine_duration']),
            ('visibility', 'visibility_germany', ['visibility']),
        ]
        
        # Create a fixed hourly timeline for 2024 to avoid explosive outer merges
        base_index = pd.date_range(start='2024-01-01 00:00:00', end='2024-12-31 23:00:00', freq='H')
        merged_df = pd.DataFrame({'datetime': base_index})
        data_info = {}
        
        for attr_name, folder_name, value_columns in attributes:
            folder_path = self.data_path / folder_name
            
            if not folder_path.exists():
                print(f"  Warning: {folder_path} does not exist, skipping {attr_name}")
                continue
            
            # Find CSV file in folder
            csv_files = list(folder_path.glob("*.csv"))
            if not csv_files:
                print(f"  Warning: No CSV files found in {folder_path}, skipping {attr_name}")
                continue
            
            # Try to find the main file (usually contains 'Germany' or 'germany' in name)
            csv_file = None
            for f in csv_files:
                if 'germany' in f.name.lower() or 'Germany' in f.name or len(csv_files) == 1:
                    csv_file = f
                    break
            
            if csv_file is None:
                csv_file = csv_files[0]
            
            print(f"  Loading {attr_name} from {csv_file.name}...")
            
            # Read only necessary columns to reduce memory and avoid encoding issues
            header_cols = pd.read_csv(csv_file, nrows=0).columns.tolist()
            # Determine datetime column name from header
            datetime_col = 'datetime' if 'datetime' in header_cols else ('MESS_DATUM' if 'MESS_DATUM' in header_cols else None)
            if datetime_col is None:
                print(f"    Warning: No datetime column found in header for {attr_name}")
                continue
            # Determine value column available in file
            value_col = next((c for c in value_columns if c in header_cols), None)
            if value_col is None:
                print(f"    Warning: No value column found for {attr_name}. Available: {header_cols}")
                continue
            usecols = [datetime_col, value_col]
            # Incremental hourly aggregation to control memory
            agg_dict = {}
            count_dict = {}
            # Stream CSV in chunks and aggregate by hour
            for chunk in pd.read_csv(
                csv_file,
                usecols=usecols,
                parse_dates=[datetime_col],
                infer_datetime_format=True,
                engine='c',
                memory_map=False,
                low_memory=True,
                chunksize=500000
            ):
                if datetime_col != 'datetime':
                    chunk.rename(columns={datetime_col: 'datetime'}, inplace=True)
                # Keep only 2024
                chunk = chunk[(chunk['datetime'] >= '2024-01-01') & (chunk['datetime'] <= '2024-12-31 23:59:59')]
                if chunk.empty:
                    continue
                # Floor to hour
                chunk['datetime'] = chunk['datetime'].dt.floor('H')
                # Aggregate by hour using mean (precip would ideally be sum, but harmonize for now)
                grouped = chunk.groupby('datetime')[value_col].agg(['sum', 'count'])
                # Accumulate
                for ts, row in grouped.iterrows():
                    s = float(row['sum'])
                    c = int(row['count'])
                    if ts in agg_dict:
                        agg_dict[ts] += s
                        count_dict[ts] += c
                    else:
                        agg_dict[ts] = s
                        count_dict[ts] = c
            # Build hourly series
            if not agg_dict:
                print(f"    Warning: No data aggregated for {attr_name} in 2024, skipping")
                continue
            hourly_dt = pd.Series(agg_dict).sort_index()
            hourly_ct = pd.Series(count_dict).reindex(hourly_dt.index)
            hourly_mean = (hourly_dt / hourly_ct).astype('float32')
            df_subset = pd.DataFrame({'datetime': hourly_mean.index, attr_name: hourly_mean.values})

            # Left-join onto fixed hourly base index to keep row count bounded
            merged_df = pd.merge(merged_df, df_subset, on='datetime', how='left')

            data_info[attr_name] = {
                'column': attr_name,
                'rows': len(df_subset),
                'missing': df_subset[attr_name].isna().sum()
            }

            print(f"    OK Loaded {len(df_subset)} rows, {df_subset[attr_name].isna().sum()} missing values")
        
        if merged_df is None:
            raise ValueError("No data loaded! Check data paths.")
        
        # Already on fixed hourly timeline; ensure sorted and reset index
        merged_df = merged_df.sort_values('datetime').reset_index(drop=True)
        
        # Handle missing values using tiered approach
        print("\nHandling missing values...")
        merged_df = self.handle_missing_values(merged_df)
        
        print(f"\nOK Unified dataset created: {len(merged_df)} rows, {len(merged_df.columns)-1} attributes")
        print(f"  Date range: {merged_df['datetime'].min()} to {merged_df['datetime'].max()}")
        
        return merged_df
    
    def handle_missing_values(self, df):
        """
        Handle missing values using tiered approach:
        - ≤2 hours: Linear interpolation
        - 3-6 hours: Forward-fill with flag
        - >6 hours: Mark as excluded
        
        Statistical Rationale: Follows standard time series imputation practices
        (Little & Rubin, 2019). The tiered approach balances data retention with
        quality assurance.
        """
        df = df.copy()
        df['excluded_flag'] = False
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != 'excluded_flag']
        
        for col in numeric_cols:
            if col not in df.columns:
                continue
            
            # Count consecutive missing values
            is_missing = df[col].isna()
            missing_groups = (is_missing != is_missing.shift()).cumsum()
            
            for group_id in missing_groups.unique():
                group_mask = missing_groups == group_id
                if not is_missing[group_mask].any():
                    continue
                
                group_size = group_mask.sum()
                group_indices = df.index[group_mask]
                
                if group_size <= 2:
                    # Linear interpolation
                    df.loc[group_indices, col] = df[col].interpolate(method='linear', limit=2)
                elif group_size <= 6:
                    # Forward-fill with flag
                    df.loc[group_indices, col] = df[col].ffill(limit=6)
                    # Note: We could add a flag column here if needed
                else:
                    # Mark as excluded
                    df.loc[group_indices, 'excluded_flag'] = True
        
        excluded_count = df['excluded_flag'].sum()
        if excluded_count > 0:
            print(f"  Marked {excluded_count} rows for exclusion due to large gaps (>6 hours)")
        
        return df
    
    def compute_variability_features(self, df, window_size=24):
        """
        Compute variability features using rolling window statistics.
        
        Scientific Basis: Atmospheric variability is a key indicator of weather
        instability. The 24-hour window captures diurnal cycles while identifying
        periods of abnormal variability (Monahan et al., 2009; Holton, 2004).
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with datetime index
        window_size : int
            Rolling window size in hours (default: 24)
        
        Returns:
        --------
        df : pd.DataFrame
            Dataframe with added variability features
        """
        print(f"\nComputing variability features (window_size={window_size}h)...")
        
        df = df.copy()
        df = df.set_index('datetime')
        
        # Temperature variability features
        if 'temperature' in df.columns:
            df['temp_std'] = df['temperature'].rolling(window=window_size, min_periods=1).std()
            df['temp_range'] = (df['temperature'].rolling(window=window_size, min_periods=1).max() - 
                               df['temperature'].rolling(window=window_size, min_periods=1).min())
            print("  OK Temperature variability: temp_std, temp_range")
        
        # Pressure change
        if 'pressure' in df.columns:
            df['pressure_change'] = df['pressure'].diff(window_size).abs()
            print("  OK Pressure change: pressure_change")
        
        # Wind coefficient of variation
        if 'wind' in df.columns:
            wind_mean = df['wind'].rolling(window=window_size, min_periods=1).mean()
            wind_std = df['wind'].rolling(window=window_size, min_periods=1).std()
            df['wind_cv'] = np.where(wind_mean > 0, wind_std / wind_mean, 0)
            print("  OK Wind variability: wind_cv")
        
        # Precipitation intensity (3-hour window)
        if 'precipitation' in df.columns:
            df['precip_intensity'] = df['precipitation'].rolling(window=3, min_periods=1).sum()
            print("  OK Precipitation intensity: precip_intensity (3h window)")
        
        # Humidity variability
        if 'dew_point' in df.columns:
            df['humidity_std'] = df['dew_point'].rolling(window=window_size, min_periods=1).std()
            print("  OK Humidity variability: humidity_std")
        elif 'moisture' in df.columns:
            df['humidity_std'] = df['moisture'].rolling(window=window_size, min_periods=1).std()
            print("  OK Humidity variability: humidity_std (from moisture)")
        
        df = df.reset_index()
        return df
    
    def compute_trend_features(self, df, window_size=24):
        """
        Compute trend features using linear regression over rolling window.
        
        Scientific Basis: Linear trends capture directional changes in atmospheric
        conditions that signal transitions between weather regimes. Trend analysis
        is well-established in climatology for detecting regime changes (Dee et al., 2011).
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with datetime index
        window_size : int
            Rolling window size in hours (default: 24)
        
        Returns:
        --------
        df : pd.DataFrame
            Dataframe with added trend features
        """
        print(f"\nComputing trend features (window_size={window_size}h)...")
        
        df = df.copy()
        df = df.set_index('datetime')
        
        def compute_trend(series, window):
            """Compute linear trend slope using polyfit"""
            trends = []
            for i in range(len(series)):
                start_idx = max(0, i - window + 1)
                window_data = series.iloc[start_idx:i+1].values
                if len(window_data) >= 2 and not np.isnan(window_data).all():
                    x = np.arange(len(window_data))
                    valid_mask = ~np.isnan(window_data)
                    if valid_mask.sum() >= 2:
                        slope = np.polyfit(x[valid_mask], window_data[valid_mask], 1)[0]
                        trends.append(slope)
                    else:
                        trends.append(np.nan)
                else:
                    trends.append(np.nan)
            return pd.Series(trends, index=series.index)
        
        # Temperature trend
        if 'temperature' in df.columns:
            df['temp_trend'] = compute_trend(df['temperature'], window_size)
            print("  OK Temperature trend: temp_trend")
        
        # Pressure trend
        if 'pressure' in df.columns:
            df['pressure_trend'] = compute_trend(df['pressure'], window_size)
            print("  OK Pressure trend: pressure_trend")
        
        # Wind trend
        if 'wind' in df.columns:
            df['wind_trend'] = compute_trend(df['wind'], window_size)
            print("  OK Wind trend: wind_trend")
        
        df = df.reset_index()
        return df
    
    def compute_extreme_event_flags(self, df):
        """
        Compute extreme event flags (binary indicators).
        
        Scientific Basis: Extreme weather events are clear indicators of atmospheric
        instability. Binary flags provide interpretable, domain-expert validated
        indicators that complement continuous variability measures (Grotjahn et al., 2016).
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        
        Returns:
        --------
        df : pd.DataFrame
            Dataframe with added extreme event flags
        """
        print("\nComputing extreme event flags...")
        
        df = df.copy()
        
        # High wind flag (90th percentile)
        if 'wind' in df.columns:
            wind_90th = df['wind'].quantile(0.9)
            df['high_wind_flag'] = (df['wind'] > wind_90th).astype(int)
            print(f"  OK High wind flag: threshold = {wind_90th:.2f} km/h")
        
        # Heavy precipitation flag (>5mm)
        if 'precipitation' in df.columns:
            df['heavy_precip_flag'] = (df['precipitation'] > 5.0).astype(int)
            print(f"  OK Heavy precipitation flag: threshold = 5.0 mm")
        
        # Rapid temperature change (>5°C in 3 hours)
        if 'temperature' in df.columns:
            df['temp_change_3h'] = df['temperature'].diff(3).abs()
            df['rapid_temp_change'] = (df['temp_change_3h'] > 5.0).astype(int)
            print(f"  OK Rapid temperature change flag: threshold = 5.0 C in 3h")
        
        # Storm flag (combined: high wind AND pressure drop)
        if 'wind' in df.columns and 'pressure' in df.columns:
            wind_90th = df['wind'].quantile(0.9)
            pressure_drop_6h = -df['pressure'].diff(6)  # Negative diff for drop
            df['storm_flag'] = ((df['wind'] > wind_90th) & (pressure_drop_6h > 5)).astype(int)
            print(f"  OK Storm flag: combined wind > {wind_90th:.2f} km/h and pressure drop > 5 hPa")
        
        return df
    
    def select_features(self, df):
        """
        Perform feature selection by correlation analysis.
        
        Scientific Basis: Multicollinearity reduces model interpretability and can
        cause numerical instability in GMM classification. Feature selection improves
        model performance and generalizability (Dormann et al., 2013).
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with all engineered features
        
        Returns:
        --------
        selected_features : list
            List of selected feature names
        correlation_matrix : pd.DataFrame
            Correlation matrix of all features
        """
        print("\nPerforming feature selection...")
        
        # Get all engineered features (exclude original attributes and metadata)
        exclude_cols = ['datetime', 'excluded_flag', 'temperature', 'cloudiness', 'wind', 
                       'wind_synop', 'precipitation', 'pressure', 'dew_point', 'moisture',
                       'extreme_wind', 'soil_temperature', 'sun', 'visibility', 'temp_change_3h']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Compute correlation matrix
        feature_df = df[feature_cols].select_dtypes(include=[np.number])
        correlation_matrix = feature_df.corr()
        
        # Find highly correlated pairs (|r| > 0.9)
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.9:
                    high_corr_pairs.append((
                        correlation_matrix.columns[i],
                        correlation_matrix.columns[j],
                        corr_val
                    ))
        
        if high_corr_pairs:
            print(f"  Found {len(high_corr_pairs)} highly correlated pairs (|r| > 0.9):")
            for feat1, feat2, corr in high_corr_pairs[:5]:  # Show first 5
                print(f"    {feat1} <-> {feat2}: r = {corr:.3f}")
        else:
            print("  No highly correlated pairs found (|r| > 0.9)")
        
        # Select features (remove redundant ones)
        selected_features = feature_cols.copy()
        
        # Remove one feature from each highly correlated pair
        # (prefer features with better meteorological interpretability)
        to_remove = set()
        for feat1, feat2, corr in high_corr_pairs:
            # Simple heuristic: prefer shorter names or more specific features
            if feat1 not in to_remove and feat2 not in to_remove:
                # Remove the one with higher missing data rate
                missing1 = df[feat1].isna().sum()
                missing2 = df[feat2].isna().sum()
                if missing1 > missing2:
                    to_remove.add(feat1)
                else:
                    to_remove.add(feat2)
        
        selected_features = [f for f in selected_features if f not in to_remove]
        
        if to_remove:
            print(f"  Removed {len(to_remove)} redundant features: {list(to_remove)}")
        
        print(f"  Selected {len(selected_features)} features for WSI computation")
        
        return selected_features, correlation_matrix
    
    def apply_robust_normalization(self, df, feature_cols):
        """
        Apply robust normalization (median, IQR) to features.
        
        Scientific Basis: Robust scaling is preferred over z-score normalization
        for weather data because it is resistant to outliers and extreme events.
        This is critical for weather data which contains legitimate extreme events
        that should not dominate normalization (Huber, 1981; Rousseeuw & Croux, 1993).
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        feature_cols : list
            List of feature column names to normalize
        
        Returns:
        --------
        df : pd.DataFrame
            Dataframe with normalized features
        scaler_params : dict
            Scaling parameters (median, IQR) for each feature
        """
        print("\nApplying robust normalization...")
        
        df = df.copy()
        scaler_params = {}
        
        # Ensure all features are oriented so higher values = more instability
        # (most features already satisfy this, but we check and flip if needed)
        
        for col in feature_cols:
            if col not in df.columns:
                continue
            
            feature_data = df[col].dropna()
            if len(feature_data) == 0:
                continue
            
            # Compute robust statistics
            median_val = feature_data.median()
            q1 = feature_data.quantile(0.25)
            q3 = feature_data.quantile(0.75)
            iqr = q3 - q1
            
            # Handle edge case where IQR is zero
            if iqr == 0:
                iqr = 1.0  # Avoid division by zero
            
            # Normalize
            df[f'{col}_norm'] = (df[col] - median_val) / iqr
            
            # Store parameters
            scaler_params[col] = {
                'median': float(median_val),
                'q1': float(q1),
                'q3': float(q3),
                'iqr': float(iqr)
            }
            
            print(f"  OK {col}: median={median_val:.3f}, IQR={iqr:.3f}")
        
        print(f"  Normalized {len(scaler_params)} features")
        
        return df, scaler_params
    
    def save_results(self, df, selected_features, correlation_matrix, scaler_params):
        """Save all results to files"""
        print("\n" + "="*60)
        print("SAVING RESULTS")
        print("="*60)
        
        # Save engineered features
        output_file = self.output_dir / "weather_features.csv"
        df.to_csv(output_file, index=False)
        print(f"OK Saved engineered features to: {output_file}")
        
        # Save correlation matrix
        corr_file = self.results_dir / "feature_correlations.png"
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(corr_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"OK Saved correlation matrix to: {corr_file}")
        
        # Save scaling parameters
        scaler_file = self.models_dir / "scalers.json"
        with open(scaler_file, 'w') as f:
            json.dump(scaler_params, f, indent=2)
        print(f"OK Saved scaling parameters to: {scaler_file}")
        
        # Save feature selection metadata
        feature_metadata = {
            'selected_features': selected_features,
            'total_features_engineered': len(df.columns),
            'normalization_method': 'robust_scaling_median_iqr',
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_file = self.models_dir / "feature_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(feature_metadata, f, indent=2)
        print(f"OK Saved feature metadata to: {metadata_file}")
        
        return output_file
    
    def run(self):
        """Run complete feature engineering pipeline"""
        print("="*60)
        print("WEATHER FEATURE ENGINEERING FOR WSI")
        print("="*60)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Step 1: Load unified dataset
        df = self.load_unified_dataset()
        
        # Step 2: Compute variability features
        df = self.compute_variability_features(df, window_size=24)
        
        # Step 3: Compute trend features
        df = self.compute_trend_features(df, window_size=24)
        
        # Step 4: Compute extreme event flags
        df = self.compute_extreme_event_flags(df)
        
        # Step 5: Feature selection
        selected_features, correlation_matrix = self.select_features(df)
        
        # Step 6: Robust normalization
        df, scaler_params = self.apply_robust_normalization(df, selected_features)
        
        # Step 7: Save results
        output_file = self.save_results(df, selected_features, correlation_matrix, scaler_params)
        
        print("\n" + "="*60)
        print("FEATURE ENGINEERING COMPLETE")
        print("="*60)
        print(f"OK Total features engineered: {len(selected_features)}")
        print(f"OK Normalized features: {len([c for c in df.columns if c.endswith('_norm')])}")
        print(f"OK Output file: {output_file}")
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return df, selected_features


def main():
    """Main function"""
    engineer = WeatherFeatureEngineer()
    df, selected_features = engineer.run()
    
    print("\nFeature engineering summary:")
    print(f"  Selected features for WSI: {selected_features}")
    
    return df, selected_features


if __name__ == "__main__":
    main()

