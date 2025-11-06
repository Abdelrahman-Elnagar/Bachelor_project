#!/usr/bin/env python3
"""
Weather Stability Index (WSI) Computation

This script implements Phase 2 of the WSI pipeline:
- Loads engineered features from Phase 1
- Computes instantaneous WSI using equal-weight averaging
- Applies 6-hour rolling window statistics
- Performs temporal smoothing using median filter
- Saves WSI timeline for classification

Scientific justification: The WSI computation follows the multi-level hierarchical
classification approach documented in the methodology framework. This approach
combines instantaneous stability metrics with temporal context to produce robust
classifications (Huth et al., 2008).

The 6-hour window is justified by synoptic-scale meteorology where weather systems
operate on characteristic timescales of 6-12 hours (Holton, 2004).

Author: Weather Stability Index Implementation
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from scipy.signal import medfilt
from scipy.stats import linregress
import warnings

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)


class WSIComputer:
    """Compute Weather Stability Index from engineered features"""
    
    def __init__(self):
        """Initialize WSI computer"""
        # Data paths
        self.features_file = Path("data/features/weather_features.csv")
        self.scalers_file = Path("models/scalers.json")
        self.feature_metadata_file = Path("models/feature_metadata.json")
        
        # Output paths
        self.output_dir = Path("data/processed")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.models_dir = Path("models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
    def load_features(self):
        """
        Load engineered features from Phase 1.
        
        Returns:
        --------
        df : pd.DataFrame
            Dataframe with engineered features
        selected_features : list
            List of selected feature names (normalized)
        """
        print("="*60)
        print("LOADING ENGINEERED FEATURES")
        print("="*60)
        
        if not self.features_file.exists():
            raise FileNotFoundError(
                f"Features file not found: {self.features_file}\n"
                "Please run feature_engineering.py first."
            )
        
        print(f"Loading features from: {self.features_file}")
        df = pd.read_csv(self.features_file)
        
        # Convert datetime
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        else:
            raise ValueError("No 'datetime' column found in features file")
        
        # Load feature metadata to get selected features
        if self.feature_metadata_file.exists():
            with open(self.feature_metadata_file, 'r') as f:
                metadata = json.load(f)
            selected_features = metadata.get('selected_features', [])
        else:
            # Fallback: find all normalized features
            selected_features = [col for col in df.columns if col.endswith('_norm')]
            print("  Warning: Using all normalized features (feature_metadata.json not found)")
        
        # Filter to normalized versions
        normalized_features = [f"{feat}_norm" for feat in selected_features if f"{feat}_norm" in df.columns]
        
        if not normalized_features:
            # Try direct matching
            normalized_features = [f for f in selected_features if f in df.columns and f.endswith('_norm')]
        
        if not normalized_features:
            raise ValueError("No normalized features found! Check feature_engineering.py output.")
        
        print(f"  Found {len(normalized_features)} normalized features")
        print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        print(f"  Total rows: {len(df)}")
        
        return df, normalized_features
    
    def compute_instantaneous_wsi(self, df, feature_cols):
        """
        Compute instantaneous WSI using equal-weight averaging.
        
        Formula: WSI_t = (1/p) * Σ[(x_i,t - median(x_i)) / IQR(x_i)]
        
        Since features are already normalized (robust scaling), this simplifies to:
        WSI_t = (1/p) * Σ(x_i,t_norm)
        
        Scientific Rationale: Equal-weight averaging is the baseline approach,
        providing interpretability and avoiding overfitting to specific features.
        The formula uses robust normalization (already applied in Phase 1) to ensure
        all features contribute equally.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataframe with normalized features
        feature_cols : list
            List of normalized feature column names
        
        Returns:
        --------
        df : pd.DataFrame
            Dataframe with added WSI_instantaneous column
        """
        print("\n" + "="*60)
        print("COMPUTING INSTANTANEOUS WSI")
        print("="*60)
        
        df = df.copy()
        
        # Select normalized features
        feature_data = df[feature_cols].copy()
        
        # Handle missing values: compute mean only on available features
        # This ensures WSI is computed even when some features are missing
        df['WSI_instantaneous'] = feature_data.mean(axis=1, skipna=True)
        
        # Count how many features were used for each row
        df['n_features_used'] = feature_data.notna().sum(axis=1)
        
        # Report statistics
        wsi_mean = df['WSI_instantaneous'].mean()
        wsi_std = df['WSI_instantaneous'].std()
        wsi_min = df['WSI_instantaneous'].min()
        wsi_max = df['WSI_instantaneous'].max()
        
        print(f"  WSI statistics:")
        print(f"    Mean: {wsi_mean:.4f}")
        print(f"    Std:  {wsi_std:.4f}")
        print(f"    Min:  {wsi_min:.4f}")
        print(f"    Max:  {wsi_max:.4f}")
        print(f"    Features used per row: {df['n_features_used'].mean():.1f} (mean)")
        
        return df
    
    def compute_rolling_window_statistics(self, df, window_size=6):
        """
        Compute rolling window statistics over 6-hour window.
        
        Scientific Basis: The 6-hour window is justified by synoptic-scale meteorology
        where weather systems operate on characteristic timescales of 6-12 hours
        (Holton, 2004). This temporal resolution captures synoptic-scale changes while
        maintaining sensitivity to mesoscale phenomena.
        
        For each time point t, uses window [t-2, t-1, t, t+1, t+2, t+3] (6 hours total).
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataframe with WSI_instantaneous
        window_size : int
            Rolling window size in hours (default: 6)
        
        Returns:
        --------
        df : pd.DataFrame
            Dataframe with added window statistics
        """
        print("\n" + "="*60)
        print(f"COMPUTING ROLLING WINDOW STATISTICS (window_size={window_size}h)")
        print("="*60)
        
        df = df.copy()
        df = df.set_index('datetime').sort_index()
        
        wsi_values = df['WSI_instantaneous'].values
        
        # Initialize arrays
        wsi_window_mean = np.full(len(df), np.nan)
        wsi_window_std = np.full(len(df), np.nan)
        wsi_window_trend = np.full(len(df), np.nan)
        
        # Compute statistics for each time point
        half_window = window_size // 2
        
        for i in range(len(df)):
            # Define window: [t-2, t-1, t, t+1, t+2, t+3] for 6-hour window
            start_idx = max(0, i - half_window)
            end_idx = min(len(df), i + half_window + 1)
            
            window_data = wsi_values[start_idx:end_idx]
            window_data = window_data[~np.isnan(window_data)]
            
            if len(window_data) > 0:
                # Mean
                wsi_window_mean[i] = np.mean(window_data)
                
                # Standard deviation
                if len(window_data) > 1:
                    wsi_window_std[i] = np.std(window_data, ddof=1)
                
                # Linear trend (slope)
                if len(window_data) >= 2:
                    x = np.arange(len(window_data))
                    try:
                        slope, intercept = np.polyfit(x, window_data, 1)
                        wsi_window_trend[i] = slope
                    except:
                        wsi_window_trend[i] = 0.0
        
        df['WSI_window_mean'] = wsi_window_mean
        df['WSI_window_std'] = wsi_window_std
        df['WSI_window_trend'] = wsi_window_trend
        
        df = df.reset_index()
        
        print(f"  Computed window statistics:")
        print(f"    WSI_window_mean: mean={df['WSI_window_mean'].mean():.4f}, std={df['WSI_window_std'].mean():.4f}")
        print(f"    WSI_window_trend: mean={df['WSI_window_trend'].mean():.4f}, std={df['WSI_window_trend'].std():.4f}")
        
        return df
    
    def apply_temporal_smoothing(self, df, kernel_size=3):
        """
        Apply temporal smoothing using median filter.
        
        Scientific Rationale: Median filter is robust to outliers and prevents
        isolated misclassifications. Kernel size of 3 (3-hour smoothing) provides
        additional noise reduction without excessive lag. Preserves sharp transitions
        while removing spurious fluctuations (Tukey, 1977).
        
        Formula: WSI_smoothed_t = median(WSI_window_mean_t-1, WSI_window_mean_t, WSI_window_mean_t+1)
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataframe with WSI_window_mean
        kernel_size : int
            Median filter kernel size (default: 3)
        
        Returns:
        --------
        df : pd.DataFrame
            Dataframe with added WSI_smoothed column
        """
        print("\n" + "="*60)
        print(f"APPLYING TEMPORAL SMOOTHING (median filter, kernel_size={kernel_size})")
        print("="*60)
        
        df = df.copy()
        
        wsi_window_mean = df['WSI_window_mean'].values
        
        # Apply median filter
        wsi_smoothed = medfilt(wsi_window_mean, kernel_size=kernel_size)
        
        df['WSI_smoothed'] = wsi_smoothed
        
        print(f"  Smoothed WSI statistics:")
        print(f"    Mean: {df['WSI_smoothed'].mean():.4f}")
        print(f"    Std:  {df['WSI_smoothed'].std():.4f}")
        print(f"    Min:  {df['WSI_smoothed'].min():.4f}")
        print(f"    Max:  {df['WSI_smoothed'].max():.4f}")
        
        # Compare with unsmoothed
        diff_mean = (df['WSI_window_mean'] - df['WSI_smoothed']).abs().mean()
        print(f"    Mean absolute difference from unsmoothed: {diff_mean:.4f}")
        
        return df
    
    def save_wsi_timeline(self, df):
        """
        Save WSI timeline to CSV file.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataframe with all WSI values
        
        Returns:
        --------
        output_file : Path
            Path to saved file
        """
        print("\n" + "="*60)
        print("SAVING WSI TIMELINE")
        print("="*60)
        
        # Select relevant columns
        output_cols = ['datetime', 'WSI_instantaneous', 'WSI_window_mean', 
                      'WSI_window_std', 'WSI_window_trend', 'WSI_smoothed', 
                      'n_features_used']
        
        # Ensure all columns exist
        available_cols = [col for col in output_cols if col in df.columns]
        
        output_df = df[available_cols].copy()
        
        output_file = self.output_dir / "wsi_timeline.csv"
        output_df.to_csv(output_file, index=False)
        
        print(f"OK Saved WSI timeline to: {output_file}")
        print(f"  Columns: {', '.join(available_cols)}")
        print(f"  Rows: {len(output_df)}")
        
        # Save WSI formula metadata
        wsi_metadata = {
            'formula': 'WSI_t = (1/p) * Σ[(x_i,t - median(x_i)) / IQR(x_i)]',
            'method': 'equal_weight_averaging',
            'normalization': 'robust_scaling_median_iqr',
            'rolling_window_size': 6,
            'smoothing_method': 'median_filter',
            'smoothing_kernel_size': 3,
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_file = self.models_dir / "wsi_formula.json"
        with open(metadata_file, 'w') as f:
            json.dump(wsi_metadata, f, indent=2)
        
        print(f"OK Saved WSI formula metadata to: {metadata_file}")
        
        return output_file
    
    def run(self):
        """Run complete WSI computation pipeline"""
        print("="*60)
        print("WEATHER STABILITY INDEX (WSI) COMPUTATION")
        print("="*60)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Step 1: Load features
        df, normalized_features = self.load_features()
        
        # Step 2: Compute instantaneous WSI
        df = self.compute_instantaneous_wsi(df, normalized_features)
        
        # Step 3: Compute rolling window statistics
        df = self.compute_rolling_window_statistics(df, window_size=6)
        
        # Step 4: Apply temporal smoothing
        df = self.apply_temporal_smoothing(df, kernel_size=3)
        
        # Step 5: Save WSI timeline
        output_file = self.save_wsi_timeline(df)
        
        print("\n" + "="*60)
        print("WSI COMPUTATION COMPLETE")
        print("="*60)
        print(f"OK Output file: {output_file}")
        print(f"OK WSI range: [{df['WSI_smoothed'].min():.4f}, {df['WSI_smoothed'].max():.4f}]")
        print(f"OK WSI mean: {df['WSI_smoothed'].mean():.4f}")
        print(f"OK WSI std: {df['WSI_smoothed'].std():.4f}")
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return df


def main():
    """Main function"""
    computer = WSIComputer()
    df = computer.run()
    
    return df


if __name__ == "__main__":
    main()

