#!/usr/bin/env python3
"""
Comprehensive Weather Data Preprocessing Script - All Attributes
Uses Imputation Instead of Row Removal

This script preprocesses weather data for all 13 attributes:
- Handles missing values (-999 → NaN)
- Removes outliers (not just flagging)
- Imputes missing values using mean (normal) or median (skewed)
- Saves preprocessed data to completely new folders

Author: Weather Data Analysis Project
"""

import pandas as pd
import numpy as np
import yaml
import os
import glob
from pathlib import Path
from datetime import datetime
import json
from scipy import stats
from scipy.stats import shapiro
import warnings
warnings.filterwarnings('ignore')

class WeatherDataPreprocessorImputation:
    def __init__(self, config_path="Scripts/preprocessing_config.yaml"):
        """Initialize preprocessor with configuration"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.missing_markers = self.config['missing_value_markers']
        self.z_threshold = self.config['outlier_detection']['z_score_threshold']
        self.iqr_multiplier = self.config['outlier_detection']['iqr_multiplier']
        self.attribute_thresholds = self.config['attribute_thresholds']
        self.validation_config = self.config['validation']
        self.attribute_mappings = self.config['attribute_column_mappings']
        
        # Create output directories (completely new folders, outside Data/)
        self.output_base = Path("Preprocessed_Data")
        self.output_bundesland_base = self.output_base / "Bundesland_aggregation"
        self.output_germany_base = self.output_base / "Germany_aggregation"
        
        # Create reports directory
        self.reports_dir = Path("Scripts/preprocessing_reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics tracking
        self.stats = {
            'files_processed': 0,
            'total_rows': 0,
            'missing_values_replaced': 0,
            'outliers_removed': 0,
            'nan_values_imputed': 0,
            'mean_imputations': 0,
            'median_imputations': 0,
            'validation_errors': []
        }
    
    def check_distribution(self, series):
        """Check if distribution is normal or skewed using Shapiro-Wilk test"""
        # Remove NaN for testing
        valid_data = series.dropna()
        
        if len(valid_data) < 3:
            return 'median'  # Default to median for very small samples
        
        # Only test if we have enough data points (Shapiro-Wilk requires 3-5000)
        if len(valid_data) > 5000:
            # Sample for efficiency
            sample = valid_data.sample(5000, random_state=42)
        elif len(valid_data) < 3:
            return 'median'
        else:
            sample = valid_data
        
        try:
            # Shapiro-Wilk test for normality
            statistic, p_value = shapiro(sample)
            # Also check skewness
            skewness = stats.skew(valid_data)
            
            # Decision: if p > 0.05 and |skewness| < 1, use mean; otherwise median
            if p_value > 0.05 and abs(skewness) < 1:
                return 'mean'
            else:
                return 'median'
        except:
            # If test fails, use median as safe default
            return 'median'
    
    def replace_missing_values(self, df, numeric_columns):
        """Replace missing value markers with NaN"""
        replaced_count = 0
        for col in numeric_columns:
            if col in df.columns:
                initial_count = df[col].isna().sum()
                # Replace all missing value markers
                for marker in self.missing_markers:
                    df[col] = df[col].replace(marker, np.nan)
                final_count = df[col].isna().sum()
                replaced_count += (final_count - initial_count)
        return replaced_count
    
    def detect_outliers(self, df, value_column, attribute_name):
        """Detect outliers using domain thresholds and statistical methods"""
        if value_column not in df.columns:
            return pd.Series(False, index=df.index)
        
        # Skip if column is all NaN
        if df[value_column].isna().all():
            return pd.Series(False, index=df.index)
        
        # Remove NaN for statistical calculations
        valid_data = df[value_column].dropna()
        if len(valid_data) == 0:
            return pd.Series(False, index=df.index)
        
        outlier_flags = pd.Series(False, index=df.index)
        
        # Method 1: Domain-specific thresholds
        threshold_flags = pd.Series(False, index=df.index)
        for attr_name, attr_config in self.attribute_thresholds.items():
            if value_column in attr_config['columns']:
                min_val = attr_config['min']
                max_val = attr_config['max']
                threshold_flags = (df[value_column] < min_val) | (df[value_column] > max_val)
                break
        
        # Method 2: Z-score method
        if len(valid_data) > 1:
            z_scores = np.abs(stats.zscore(valid_data))
            z_flags = pd.Series(False, index=df.index)
            z_flags.loc[valid_data.index] = z_scores > self.z_threshold
        else:
            z_flags = pd.Series(False, index=df.index)
        
        # Method 3: IQR method
        if len(valid_data) > 3:
            Q1 = valid_data.quantile(0.25)
            Q3 = valid_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.iqr_multiplier * IQR
            upper_bound = Q3 + self.iqr_multiplier * IQR
            iqr_flags = (df[value_column] < lower_bound) | (df[value_column] > upper_bound)
        else:
            iqr_flags = pd.Series(False, index=df.index)
        
        # Combine all flags (OR operation)
        combined_flags = threshold_flags | z_flags | iqr_flags
        
        return combined_flags
    
    def impute_missing_values(self, df, value_column):
        """Impute missing values using mean (normal) or median (skewed)"""
        if value_column not in df.columns:
            return 0, 0, 0
        
        nan_before = df[value_column].isna().sum()
        if nan_before == 0:
            return 0, 0, 0
        
        # Check distribution
        dist_type = self.check_distribution(df[value_column])
        
        # Get valid data for imputation
        valid_data = df[value_column].dropna()
        
        if len(valid_data) == 0:
            # If all data is missing, can't impute - set to 0 or keep NaN
            return 0, 0, nan_before
        
        # Impute based on distribution
        if dist_type == 'mean':
            impute_value = valid_data.mean()
            df[value_column].fillna(impute_value, inplace=True)
            return nan_before, nan_before, 0
        else:
            impute_value = valid_data.median()
            df[value_column].fillna(impute_value, inplace=True)
            return nan_before, 0, nan_before
    
    def process_file(self, file_path, output_folder, attribute_name, value_columns):
        """Process a single CSV file: remove outliers and impute missing values"""
        try:
            # Read CSV with proper quote handling for fields with commas
            # Use quotechar and proper escaping for the contributing_stations column
            try:
                df = pd.read_csv(file_path, quotechar='"', 
                                on_bad_lines='skip', engine='python', encoding='utf-8')
            except Exception as e1:
                # Try with different encoding
                try:
                    df = pd.read_csv(file_path, quotechar='"', 
                                    on_bad_lines='skip', engine='python', encoding='latin-1')
                except Exception as e2:
                    # Fallback: try C engine (faster) for problematic files
                    try:
                        df = pd.read_csv(file_path, low_memory=False, on_bad_lines='skip')
                    except:
                        # Last resort: python engine without low_memory
                        print(f"    Warning: Standard CSV reading failed for {file_path.name}, trying alternative method...")
                        df = pd.read_csv(file_path, on_bad_lines='skip', 
                                        engine='python', sep=',', quoting=1)
            
            initial_rows = len(df)
            
            # Check if file was already processed (avoid duplicates) - check before processing
            output_path = output_folder / file_path.name
            if output_path.exists():
                # Try to verify it's complete by checking file size and row count
                try:
                    existing_df = pd.read_csv(output_path, on_bad_lines='skip', engine='python')
                    # Quick check: if output has reasonable row count (at least 80% of original), consider it done
                    if len(existing_df) >= int(initial_rows * 0.8):
                        print(f"    Skipping {file_path.name} - already processed ({len(existing_df)} rows)")
                        return None
                    else:
                        print(f"    Re-processing {file_path.name} - incomplete previous run ({len(existing_df)} < {initial_rows})")
                except Exception as e:
                    print(f"    Re-processing {file_path.name} - corrupted output file ({str(e)})")
            
            # Ensure datetime column exists
            datetime_col = 'datetime' if 'datetime' in df.columns else 'MESS_DATUM'
            if datetime_col != 'datetime':
                if datetime_col in df.columns:
                    df.rename(columns={datetime_col: 'datetime'}, inplace=True)
            
            # Convert datetime if needed
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            
            # Find value column
            value_column = None
            for col in value_columns:
                if col in df.columns:
                    value_column = col
                    break
            
            if value_column is None:
                print(f"  Warning: No value column found in {file_path.name}")
                return None
            
            # Identify numeric columns
            exclude_cols = ['STATIONS_ID', 'station_id', 'stations', 'STATIONS_CONTRIBUTING', 
                          'datetime', 'MESS_DATUM', 'outlier_flag', 'contributing_stations']
            numeric_cols = [col for col in df.columns 
                          if col not in exclude_cols and 
                          (df[col].dtype in ['int64', 'float64'] or 
                           pd.to_numeric(df[col], errors='coerce').notna().any())]
            
            # Step 1: Replace missing values
            missing_replaced = self.replace_missing_values(df, numeric_cols)
            
            # Step 2: Detect outliers
            outlier_flags = self.detect_outliers(df, value_column, attribute_name)
            outliers_count = outlier_flags.sum()
            
            # Step 3: Remove outliers (set to NaN first, then impute)
            df.loc[outlier_flags, value_column] = np.nan
            
            # Step 4: Impute missing values (including removed outliers)
            imputed_count, mean_imputations, median_imputations = self.impute_missing_values(df, value_column)
            
            # Step 5: Remove outlier_flag column if it exists
            if 'outlier_flag' in df.columns:
                df = df.drop(columns=['outlier_flag'])
            
            final_rows = len(df)
            
            # Save to output folder
            df.to_csv(output_path, index=False, quoting=1)  # quoting=1 ensures proper quote handling
            
            # Update statistics
            self.stats['files_processed'] += 1
            self.stats['total_rows'] += final_rows
            self.stats['missing_values_replaced'] += missing_replaced
            self.stats['outliers_removed'] += outliers_count
            self.stats['nan_values_imputed'] += imputed_count
            self.stats['mean_imputations'] += mean_imputations
            self.stats['median_imputations'] += median_imputations
            
            file_report = {
                'file': str(file_path),
                'output_file': str(output_path),
                'attribute': attribute_name,
                'rows': final_rows,
                'missing_replaced': int(missing_replaced),
                'outliers_removed': int(outliers_count),
                'nan_imputed': int(imputed_count),
                'mean_imputations': int(mean_imputations),
                'median_imputations': int(median_imputations)
            }
            
            return file_report
            
        except Exception as e:
            error_report = {
                'file': str(file_path),
                'error': str(e),
                'attribute': attribute_name
            }
            self.stats['validation_errors'].append(error_report)
            print(f"  Error processing {file_path.name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_attribute(self, attribute_name, bundesland_folder, germany_folder):
        """Process all files for a single attribute"""
        base_path = Path("Data")
        all_reports = []
        
        # Get value columns from config
        if attribute_name not in self.attribute_mappings:
            print(f"  Warning: Attribute {attribute_name} not found in config")
            return []
        
        attr_config = self.attribute_mappings[attribute_name]
        bundesland_folder_name = attr_config['bundesland_folder']
        germany_folder_name = attr_config['germany_folder']
        value_columns = attr_config['value_columns']
        
        # Create output folders
        output_bundesland = self.output_bundesland_base / bundesland_folder_name
        output_germany = self.output_germany_base / germany_folder_name
        output_bundesland.mkdir(parents=True, exist_ok=True)
        output_germany.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Processing {attribute_name}")
        print(f"{'='*60}")
        
        # Process Bundesland folder
        bundesland_path = base_path / "Bundesland_aggregation" / bundesland_folder_name
        if bundesland_path.exists():
            csv_files = list(bundesland_path.glob("*.csv"))
            print(f"  Bundesland folder: {len(csv_files)} files")
            for file_path in csv_files:
                if '_SUMMARY' in file_path.name.upper() or '_AGGREGATION' in file_path.name.upper():
                    continue
                try:
                    report = self.process_file(file_path, output_bundesland, attribute_name, value_columns)
                    if report:
                        all_reports.append(report)
                        print(f"    {file_path.name}: {report['outliers_removed']} outliers, {report['nan_imputed']} imputed")
                except Exception as e:
                    error_msg = str(e)[:100]  # Truncate long error messages
                    print(f"    ERROR processing {file_path.name}: {error_msg}")
                    print(f"    Continuing with next file...")
                    continue
        else:
            print(f"  Warning: Bundesland folder not found: {bundesland_path}")
        
        # Process Germany folder
        germany_path = base_path / "Germany_aggregation" / germany_folder_name
        if germany_path.exists():
            csv_files = list(germany_path.glob("*.csv"))
            print(f"  Germany folder: {len(csv_files)} files")
            for file_path in csv_files:
                if '_SUMMARY' in file_path.name.upper() or '_AGGREGATION' in file_path.name.upper():
                    continue
                try:
                    report = self.process_file(file_path, output_germany, attribute_name, value_columns)
                    if report:
                        all_reports.append(report)
                        print(f"    {file_path.name}: {report['outliers_removed']} outliers, {report['nan_imputed']} imputed")
                except Exception as e:
                    error_msg = str(e)[:100]  # Truncate long error messages
                    print(f"    ERROR processing {file_path.name}: {error_msg}")
                    print(f"    Continuing with next file...")
                    continue
        else:
            print(f"  Warning: Germany folder not found: {germany_path}")
        
        return all_reports
    
    def generate_summary_report(self, all_reports):
        """Generate summary report"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        report_file = self.reports_dir / f"preprocessing_imputation_{timestamp}.json"
        
        summary = {
            'timestamp': timestamp,
            'overall_stats': self.stats,
            'file_reports': all_reports,
            'config': {
                'missing_value_markers': self.missing_markers,
                'z_score_threshold': self.z_threshold,
                'iqr_multiplier': self.iqr_multiplier
            }
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Also create a CSV summary
        csv_report_file = self.reports_dir / f"preprocessing_imputation_{timestamp}.csv"
        if all_reports:
            summary_df = pd.DataFrame(all_reports)
            summary_df.to_csv(csv_report_file, index=False)
        
        print(f"\n{'='*60}")
        print("PREPROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Files processed: {self.stats['files_processed']}")
        print(f"Total rows: {self.stats['total_rows']:,}")
        print(f"Missing values replaced: {self.stats['missing_values_replaced']:,}")
        print(f"Outliers removed: {self.stats['outliers_removed']:,}")
        print(f"NaN values imputed: {self.stats['nan_values_imputed']:,}")
        print(f"  - Mean imputations: {self.stats['mean_imputations']:,}")
        print(f"  - Median imputations: {self.stats['median_imputations']:,}")
        print(f"Validation errors: {len(self.stats['validation_errors'])}")
        print(f"\nReports saved to: {report_file}")
        print(f"CSV summary saved to: {csv_report_file}")
        print(f"\nPreprocessed data saved to: {self.output_base}")
        
        return summary


def main():
    """Main preprocessing function"""
    # List of all 13 attributes
    attributes = [
        ('wind_synop', 'Bundesland_aggregation/wind_synop_by_bundesland', 'Germany_aggregation/wind_synop_germany'),
        ('precipitation', 'Bundesland_aggregation/precipitation_by_bundesland', 'Germany_aggregation/precipitation_germany'),
        ('temperature', 'Bundesland_aggregation/Temp_Bundesland_Aggregated', 'Germany_aggregation/Temp_Germany_Aggregated'),
        ('cloudiness', 'Bundesland_aggregation/Cloudness_Bundesland_Aggregated', 'Germany_aggregation/Cloudness_Germany_Aggregated'),
        ('dew_point', 'Bundesland_aggregation/dew_point_by_bundesland', 'Germany_aggregation/dew_point_germany_aggregated'),
        ('extreme_wind', 'Bundesland_aggregation/extreme_wind_by_bundesland', 'Germany_aggregation/extreme_wind_germany_aggregated'),
        ('moisture', 'Bundesland_aggregation/moisture_by_bundesland', 'Germany_aggregation/moisture_germany_aggregated'),
        ('pressure', 'Bundesland_aggregation/pressure_by_bundesland', 'Germany_aggregation/pressure_germany'),
        ('soil_temperature', 'Bundesland_aggregation/soil_temperature_by_bundesland', 'Germany_aggregation/soil_temperature_germany'),
        ('sun', 'Bundesland_aggregation/sun_by_bundesland', 'Germany_aggregation/sun_germany'),
        ('visibility', 'Bundesland_aggregation/visibility_by_bundesland', 'Germany_aggregation/visibility_germany'),
        ('wind', 'Bundesland_aggregation/wind_by_bundesland', 'Germany_aggregation/wind_germany'),
    ]
    
    preprocessor = WeatherDataPreprocessorImputation()
    all_reports = []
    
    print("="*60)
    print("WEATHER DATA PREPROCESSING - ALL ATTRIBUTES")
    print("IMPUTATION METHOD (NO ROW REMOVAL)")
    print("="*60)
    print("This script will:")
    print("  1. Replace missing value markers with NaN")
    print("  2. Remove outliers (replace with NaN, then impute)")
    print("  3. Impute missing values using:")
    print("     - Mean: if distribution is normal (Shapiro-Wilk p > 0.05, |skewness| < 1)")
    print("     - Median: if distribution is skewed")
    print("  4. Save preprocessed data to: Preprocessed_Data/")
    print("="*60)
    
    for attr_name, bundesland_folder, germany_folder in attributes:
        try:
            reports = preprocessor.process_attribute(attr_name, bundesland_folder, germany_folder)
            all_reports.extend(reports)
        except Exception as e:
            print(f"Error processing {attr_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate summary
    if all_reports:
        preprocessor.generate_summary_report(all_reports)
    
    print("\nPreprocessing complete!")


if __name__ == "__main__":
    main()

