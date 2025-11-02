#!/usr/bin/env python3
"""
Comprehensive Weather Data Preprocessing Script

This script preprocesses weather data files in-place:
- Handles missing values (-999 → NaN)
- Flags outliers (does not remove them)
- Validates data types and timestamps
- Detects duplicates
- Generates preprocessing reports

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
import warnings
warnings.filterwarnings('ignore')

class WeatherDataPreprocessor:
    def __init__(self, config_path="Scripts/preprocessing_config.yaml"):
        """Initialize preprocessor with configuration"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.missing_markers = self.config['missing_value_markers']
        self.z_threshold = self.config['outlier_detection']['z_score_threshold']
        self.iqr_multiplier = self.config['outlier_detection']['iqr_multiplier']
        self.attribute_thresholds = self.config['attribute_thresholds']
        self.validation_config = self.config['validation']
        
        # Create reports directory
        self.reports_dir = Path("Scripts/preprocessing_reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics tracking
        self.stats = {
            'files_processed': 0,
            'total_rows': 0,
            'missing_values_replaced': 0,
            'outliers_flagged': 0,
            'duplicates_found': 0,
            'validation_errors': []
        }
    
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
    
    def flag_outliers(self, df, numeric_columns):
        """Flag outliers using domain thresholds and statistical methods"""
        outlier_count = 0
        
        # Initialize outlier flag column if it doesn't exist
        if 'outlier_flag' not in df.columns:
            df['outlier_flag'] = False
        
        for col in numeric_columns:
            if col not in df.columns:
                continue
                
            # Skip if column is all NaN
            if df[col].isna().all():
                continue
            
            # Remove NaN for statistical calculations
            valid_data = df[col].dropna()
            if len(valid_data) == 0:
                continue
            
            # Method 1: Domain-specific thresholds
            threshold_flags = False
            for attr_name, attr_config in self.attribute_thresholds.items():
                if col in attr_config['columns']:
                    min_val = attr_config['min']
                    max_val = attr_config['max']
                    threshold_flags = (df[col] < min_val) | (df[col] > max_val)
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
                iqr_flags = (df[col] < lower_bound) | (df[col] > upper_bound)
            else:
                iqr_flags = pd.Series(False, index=df.index)
            
            # Combine all flags (OR operation)
            if isinstance(threshold_flags, pd.Series):
                combined_flags = threshold_flags | z_flags | iqr_flags
            else:
                combined_flags = z_flags | iqr_flags
            
            df['outlier_flag'] = df['outlier_flag'] | combined_flags
            outlier_count += combined_flags.sum()
        
        return outlier_count
    
    def validate_data_types(self, df):
        """Validate and convert data types"""
        errors = []
        
        # Validate timestamp column
        if self.validation_config['timestamp_column'] in df.columns:
            try:
                df[self.validation_config['timestamp_column']] = pd.to_datetime(
                    df[self.validation_config['timestamp_column']]
                )
            except Exception as e:
                errors.append(f"Timestamp conversion error: {e}")
        
        # Identify numeric columns and convert
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if col not in [self.validation_config['timestamp_column']]:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception as e:
                    errors.append(f"Numeric conversion error for {col}: {e}")
        
        return errors
    
    def validate_timestamps(self, df):
        """Validate timestamps are in expected year"""
        errors = []
        timestamp_col = self.validation_config['timestamp_column']
        expected_year = self.validation_config['expected_year']
        
        if timestamp_col in df.columns:
            # Check if timestamps are in expected year
            if df[timestamp_col].dtype == 'datetime64[ns]':
                invalid_years = df[df[timestamp_col].dt.year != expected_year]
                if len(invalid_years) > 0:
                    errors.append(
                        f"Found {len(invalid_years)} rows with year != {expected_year}"
                    )
        
        return errors
    
    def detect_duplicates(self, df, timestamp_col=None):
        """Detect duplicate records"""
        duplicates = 0
        # Try both old and new column names
        if timestamp_col is None:
            timestamp_col = 'datetime' if 'datetime' in df.columns else 'MESS_DATUM'
        
        if timestamp_col in df.columns:
            # Check for duplicate timestamps
            duplicates = df.duplicated(subset=[timestamp_col], keep='first').sum()
        return duplicates
    
    def calculate_completeness(self, df, value_columns):
        """Calculate data completeness percentage"""
        if len(value_columns) == 0:
            return 0.0
        
        total_cells = len(df) * len(value_columns)
        if total_cells == 0:
            return 0.0
        
        non_null_cells = 0
        for col in value_columns:
            if col in df.columns:
                non_null_cells += df[col].notna().sum()
        
        return (non_null_cells / total_cells) * 100
    
    def process_file(self, file_path, attribute_name=None):
        """Process a single CSV file"""
        try:
            # Read CSV
            df = pd.read_csv(file_path, low_memory=False)
            initial_rows = len(df)
            
            # Identify numeric columns (exclude metadata columns)
            exclude_cols = ['STATIONS_ID', 'station_id', 'stations', 'STATIONS_CONTRIBUTING', 
                          'MESS_DATUM', 'outlier_flag']
            numeric_cols = [col for col in df.columns 
                          if col not in exclude_cols and 
                          df[col].dtype in ['int64', 'float64', 'object']]
            
            # Filter numeric columns that can actually be numeric
            numeric_cols = [col for col in numeric_cols 
                          if df[col].dtype in ['int64', 'float64'] or 
                          pd.to_numeric(df[col], errors='coerce').notna().any()]
            
            # Step 1: Replace missing values
            missing_replaced = self.replace_missing_values(df, numeric_cols)
            
            # Step 2: Validate data types
            type_errors = self.validate_data_types(df)
            
            # Step 3: Validate timestamps
            timestamp_errors = self.validate_timestamps(df)
            
            # Step 4: Detect duplicates
            duplicates = self.detect_duplicates(df)
            
            # Step 5: Flag outliers
            outliers_flagged = self.flag_outliers(df, numeric_cols)
            
            # Step 6: Calculate completeness
            completeness = self.calculate_completeness(df, numeric_cols)
            
            # Save processed file in-place
            df.to_csv(file_path, index=False)
            
            # Update statistics
            self.stats['files_processed'] += 1
            self.stats['total_rows'] += initial_rows
            self.stats['missing_values_replaced'] += missing_replaced
            self.stats['outliers_flagged'] += outliers_flagged
            self.stats['duplicates_found'] += duplicates
            
            file_report = {
                'file': str(file_path),
                'attribute': attribute_name or 'unknown',
                'rows': initial_rows,
                'missing_replaced': int(missing_replaced),
                'outliers_flagged': int(outliers_flagged),
                'duplicates': int(duplicates),
                'completeness': round(completeness, 2),
                'type_errors': type_errors,
                'timestamp_errors': timestamp_errors
            }
            
            return file_report
            
        except Exception as e:
            error_report = {
                'file': str(file_path),
                'error': str(e),
                'attribute': attribute_name or 'unknown'
            }
            self.stats['validation_errors'].append(error_report)
            return None
    
    def process_attribute_folder(self, folder_path, attribute_name):
        """Process all CSV files in an attribute folder"""
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        reports = []
        
        print(f"\nProcessing {attribute_name} in {folder_path}...")
        print(f"Found {len(csv_files)} CSV files")
        
        for file_path in csv_files:
            # Skip summary files
            if '_SUMMARY' in os.path.basename(file_path).upper() or \
               '_AGGREGATION' in os.path.basename(file_path).upper():
                continue
            
            report = self.process_file(file_path, attribute_name)
            if report:
                reports.append(report)
                print(f"  Processed: {os.path.basename(file_path)} - "
                      f"{report['rows']} rows, {report['missing_replaced']} missing replaced, "
                      f"{report['outliers_flagged']} outliers flagged")
        
        return reports
    
    def generate_summary_report(self, all_reports):
        """Generate summary report"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        report_file = self.reports_dir / f"preprocessing_summary_{timestamp}.json"
        
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
        csv_report_file = self.reports_dir / f"preprocessing_summary_{timestamp}.csv"
        if all_reports:
            summary_df = pd.DataFrame(all_reports)
            summary_df.to_csv(csv_report_file, index=False)
        
        print(f"\n{'='*60}")
        print("PREPROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Files processed: {self.stats['files_processed']}")
        print(f"Total rows: {self.stats['total_rows']:,}")
        print(f"Missing values replaced: {self.stats['missing_values_replaced']:,}")
        print(f"Outliers flagged: {self.stats['outliers_flagged']:,}")
        print(f"Duplicates found: {self.stats['duplicates_found']:,}")
        print(f"Validation errors: {len(self.stats['validation_errors'])}")
        print(f"\nReports saved to: {report_file}")
        print(f"CSV summary saved to: {csv_report_file}")
        
        return summary


def main():
    """Main preprocessing function"""
    base_path = Path("Data")
    
    # List of attribute folders to process
    # Format: (attribute_name, bundesland_folder, germany_folder)
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
    
    preprocessor = WeatherDataPreprocessor()
    all_reports = []
    
    print("="*60)
    print("WEATHER DATA PREPROCESSING")
    print("="*60)
    print(f"Processing files in-place...")
    
    for attr_name, bundesland_folder, germany_folder in attributes:
        # Process Bundesland folder
        bundesland_path = base_path / bundesland_folder
        if bundesland_path.exists():
            reports = preprocessor.process_attribute_folder(bundesland_path, attr_name)
            all_reports.extend(reports)
        else:
            print(f"Folder not found: {bundesland_path}")
        
        # Process Germany folder
        germany_path = base_path / germany_folder
        if germany_path.exists():
            reports = preprocessor.process_attribute_folder(germany_path, attr_name)
            all_reports.extend(reports)
        else:
            print(f"Folder not found: {germany_path}")
    
    # Generate summary
    preprocessor.generate_summary_report(all_reports)
    
    print("\nPreprocessing complete!")


if __name__ == "__main__":
    main()

