#!/usr/bin/env python3
"""
Temperature Data Preprocessing Script - Remove Outliers and NaN

This script preprocesses temperature data by:
- Handling missing values (-999 → NaN)
- Removing outliers (not just flagging)
- Removing rows with NaN values
- Saving preprocessed data to a new folder (not in-place)

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

class TemperaturePreprocessor:
    def __init__(self, config_path="Scripts/preprocessing_config.yaml"):
        """Initialize preprocessor with configuration"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.missing_markers = self.config['missing_value_markers']
        self.z_threshold = self.config['outlier_detection']['z_score_threshold']
        self.iqr_multiplier = self.config['outlier_detection']['iqr_multiplier']
        self.temperature_thresholds = self.config['attribute_thresholds']['temperature']
        
        # Create output directories
        self.output_base = Path("Data")
        self.output_bundesland = self.output_base / "Bundesland_aggregation" / "Temp_Bundesland_Aggregated_preprocessed"
        self.output_germany = self.output_base / "Germany_aggregation" / "Temp_Germany_Aggregated_preprocessed"
        self.output_bundesland.mkdir(parents=True, exist_ok=True)
        self.output_germany.mkdir(parents=True, exist_ok=True)
        
        # Create reports directory
        self.reports_dir = Path("Scripts/preprocessing_reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics tracking
        self.stats = {
            'files_processed': 0,
            'total_rows_before': 0,
            'total_rows_after': 0,
            'missing_values_replaced': 0,
            'outliers_removed': 0,
            'nan_rows_removed': 0,
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
    
    def detect_outliers(self, df, temperature_column):
        """Detect outliers in temperature column using multiple methods"""
        if temperature_column not in df.columns:
            return pd.Series(False, index=df.index)
        
        # Skip if column is all NaN
        if df[temperature_column].isna().all():
            return pd.Series(False, index=df.index)
        
        # Remove NaN for statistical calculations
        valid_data = df[temperature_column].dropna()
        if len(valid_data) == 0:
            return pd.Series(False, index=df.index)
        
        outlier_flags = pd.Series(False, index=df.index)
        
        # Method 1: Domain-specific thresholds
        min_val = self.temperature_thresholds['min']
        max_val = self.temperature_thresholds['max']
        threshold_flags = (df[temperature_column] < min_val) | (df[temperature_column] > max_val)
        
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
            iqr_flags = (df[temperature_column] < lower_bound) | (df[temperature_column] > upper_bound)
        else:
            iqr_flags = pd.Series(False, index=df.index)
        
        # Combine all flags (OR operation)
        combined_flags = threshold_flags | z_flags | iqr_flags
        
        return combined_flags
    
    def process_file(self, file_path, output_folder, attribute_name="temperature"):
        """Process a single CSV file: remove outliers and NaN"""
        try:
            # Read CSV
            df = pd.read_csv(file_path, low_memory=False)
            initial_rows = len(df)
            
            # Ensure datetime column exists
            datetime_col = 'datetime' if 'datetime' in df.columns else 'MESS_DATUM'
            if datetime_col != 'datetime':
                if datetime_col in df.columns:
                    df.rename(columns={datetime_col: 'datetime'}, inplace=True)
            
            # Convert datetime if needed
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            
            # Identify temperature column
            temperature_columns = ['temperature_mean', 'TT_TU_mean', 'TT', 'TT_STD']
            temperature_column = None
            for col in temperature_columns:
                if col in df.columns:
                    temperature_column = col
                    break
            
            if temperature_column is None:
                print(f"  Warning: No temperature column found in {file_path.name}")
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
            outlier_flags = self.detect_outliers(df, temperature_column)
            outliers_count = outlier_flags.sum()
            
            # Step 3: Remove outliers
            df_cleaned = df[~outlier_flags].copy()
            
            # Step 4: Remove rows with NaN in temperature column
            nan_before = df_cleaned[temperature_column].isna().sum()
            df_cleaned = df_cleaned.dropna(subset=[temperature_column])
            nan_removed = nan_before
            
            # Step 5: Remove outlier_flag column (no longer needed)
            if 'outlier_flag' in df_cleaned.columns:
                df_cleaned = df_cleaned.drop(columns=['outlier_flag'])
            
            final_rows = len(df_cleaned)
            
            # Save to output folder
            output_path = output_folder / file_path.name
            df_cleaned.to_csv(output_path, index=False)
            
            # Update statistics
            self.stats['files_processed'] += 1
            self.stats['total_rows_before'] += initial_rows
            self.stats['total_rows_after'] += final_rows
            self.stats['missing_values_replaced'] += missing_replaced
            self.stats['outliers_removed'] += outliers_count
            self.stats['nan_rows_removed'] += nan_removed
            
            file_report = {
                'file': str(file_path),
                'output_file': str(output_path),
                'attribute': attribute_name,
                'rows_before': initial_rows,
                'rows_after': final_rows,
                'rows_removed': initial_rows - final_rows,
                'missing_replaced': int(missing_replaced),
                'outliers_removed': int(outliers_count),
                'nan_rows_removed': int(nan_removed),
                'data_loss_percent': round((1 - final_rows / initial_rows) * 100, 2) if initial_rows > 0 else 0
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
            return None
    
    def process_folder(self, folder_path, output_folder, attribute_name="temperature"):
        """Process all CSV files in a folder"""
        csv_files = list(folder_path.glob("*.csv"))
        reports = []
        
        print(f"\nProcessing {attribute_name} in {folder_path}...")
        print(f"Found {len(csv_files)} CSV files")
        print(f"Output folder: {output_folder}")
        
        for file_path in csv_files:
            # Skip summary files
            if '_SUMMARY' in file_path.name.upper() or \
               '_AGGREGATION' in file_path.name.upper():
                continue
            
            report = self.process_file(file_path, output_folder, attribute_name)
            if report:
                reports.append(report)
                print(f"  Processed: {file_path.name} - "
                      f"{report['rows_before']} -> {report['rows_after']} rows "
                      f"({report['data_loss_percent']:.2f}% removed), "
                      f"{report['outliers_removed']} outliers, "
                      f"{report['nan_rows_removed']} NaN rows removed")
        
        return reports
    
    def generate_summary_report(self, all_reports):
        """Generate summary report"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        report_file = self.reports_dir / f"temperature_preprocessing_{timestamp}.json"
        
        summary = {
            'timestamp': timestamp,
            'overall_stats': self.stats,
            'file_reports': all_reports,
            'config': {
                'missing_value_markers': self.missing_markers,
                'z_score_threshold': self.z_threshold,
                'iqr_multiplier': self.iqr_multiplier,
                'temperature_thresholds': self.temperature_thresholds
            }
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Also create a CSV summary
        csv_report_file = self.reports_dir / f"temperature_preprocessing_{timestamp}.csv"
        if all_reports:
            summary_df = pd.DataFrame(all_reports)
            summary_df.to_csv(csv_report_file, index=False)
        
        print(f"\n{'='*60}")
        print("TEMPERATURE PREPROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Files processed: {self.stats['files_processed']}")
        print(f"Total rows before: {self.stats['total_rows_before']:,}")
        print(f"Total rows after: {self.stats['total_rows_after']:,}")
        print(f"Rows removed: {self.stats['total_rows_before'] - self.stats['total_rows_after']:,}")
        print(f"Data loss: {((self.stats['total_rows_before'] - self.stats['total_rows_after']) / self.stats['total_rows_before'] * 100):.2f}%" if self.stats['total_rows_before'] > 0 else "0%")
        print(f"Missing values replaced: {self.stats['missing_values_replaced']:,}")
        print(f"Outliers removed: {self.stats['outliers_removed']:,}")
        print(f"NaN rows removed: {self.stats['nan_rows_removed']:,}")
        print(f"Validation errors: {len(self.stats['validation_errors'])}")
        print(f"\nReports saved to: {report_file}")
        print(f"CSV summary saved to: {csv_report_file}")
        print(f"\nPreprocessed data saved to:")
        print(f"  - {self.output_bundesland}")
        print(f"  - {self.output_germany}")
        
        return summary


def main():
    """Main preprocessing function"""
    base_path = Path("Data")
    
    bundesland_folder = base_path / "Bundesland_aggregation" / "Temp_Bundesland_Aggregated"
    germany_folder = base_path / "Germany_aggregation" / "Temp_Germany_Aggregated"
    
    preprocessor = TemperaturePreprocessor()
    all_reports = []
    
    print("="*60)
    print("TEMPERATURE DATA PREPROCESSING - REMOVE OUTLIERS & NaN")
    print("="*60)
    print("This script will:")
    print("  1. Replace missing value markers with NaN")
    print("  2. Remove outliers using domain thresholds, Z-score, and IQR methods")
    print("  3. Remove rows with NaN values in temperature column")
    print("  4. Save preprocessed data to new folders (not in-place)")
    print("="*60)
    
    # Process Bundesland folder
    if bundesland_folder.exists():
        reports = preprocessor.process_folder(bundesland_folder, 
                                            preprocessor.output_bundesland, 
                                            "temperature")
        all_reports.extend(reports)
    else:
        print(f"Folder not found: {bundesland_folder}")
    
    # Process Germany folder
    if germany_folder.exists():
        reports = preprocessor.process_folder(germany_folder, 
                                            preprocessor.output_germany, 
                                            "temperature")
        all_reports.extend(reports)
    else:
        print(f"Folder not found: {germany_folder}")
    
    # Generate summary
    if all_reports:
        preprocessor.generate_summary_report(all_reports)
    
    print("\nPreprocessing complete!")


if __name__ == "__main__":
    main()

