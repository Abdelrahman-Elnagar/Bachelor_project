#!/usr/bin/env python3
"""
Energy Data Ingestion and Preprocessing for Renewable Energy Forecasting

This script implements comprehensive data acquisition, validation, and preprocessing
for solar PV and wind power generation data (Germany, 2024).

Features:
- Multi-format support (API, CSV, JSON)
- Comprehensive validation (completeness, outliers, alignment)
- Automatic problem detection (gaps, anomalies, misalignments)
- Preprocessing pipeline (missing values, timezone, feature engineering)
- Quality reporting

Author: Energy Forecasting Pipeline
"""

import pandas as pd
import numpy as np
import requests
import yaml
import json
from pathlib import Path
from datetime import datetime, timezone
import warnings
from typing import Dict, List, Optional, Tuple
import logging

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)


class EnergyDataIngester:
    """Ingest and preprocess renewable energy production data"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize energy data ingester
        
        Parameters:
        -----------
        config_path : str, optional
            Path to configuration file (YAML). If None, uses defaults.
        """
        # Set up directories (relative to forecasting/)
        self.base_dir = Path(__file__).parent.parent
        self.data_dir = self.base_dir / "data"
        self.processed_dir = self.data_dir / "processed"
        self.raw_dir = self.data_dir / "raw"
        self.config_dir = self.data_dir / "config"
        self.results_dir = self.base_dir / "results"
        self.logs_dir = self.base_dir / "logs"
        
        # Create directories
        for dir_path in [self.processed_dir, self.raw_dir, self.config_dir, 
                        self.results_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        log_file = self.logs_dir / f"energy_ingestion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        if config_path is None:
            config_path = self.config_dir / "forecasting.yaml"
        
        self.config = self._load_config(config_path)
        
        # Reference data (WSI timeline from parent directory)
        # Path relative to forecasting/ directory
        self.wsi_labels_path = Path(__file__).parent.parent.parent / "Data" / "processed" / "stability_labels.csv"
        self.expected_hours_2024 = 8784  # 2024 is a leap year
        
        # Validation thresholds
        self.validation_config = {
            'max_missing_pct': 5.0,  # Max 5% missing data
            'outlier_z_threshold': 4.0,  # Z-score threshold for outliers
            'min_capacity_pct': 0.0,  # Minimum generation (0% of capacity)
            'max_capacity_pct': 1.2,  # Maximum generation (120% of capacity, allows overproduction)
            'min_solar_capacity_mw': 0,  # Minimum solar capacity (will be updated)
            'max_solar_capacity_mw': 70000,  # Approx. 70 GW installed in Germany 2024
            'min_wind_capacity_mw': 0,
            'max_wind_capacity_mw': 70000,  # Approx. 70 GW installed in Germany 2024
        }
        
    def _load_config(self, config_path: Path) -> Dict:
        """Load configuration file"""
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            # Return default configuration
            self.logger.warning(f"Config file not found: {config_path}. Using defaults.")
            return {
                'data_sources': {
                    'entsoe': {
                        'enabled': False,
                        'api_key': None,
                        'base_url': 'https://web-api.tp.entsoe.eu/api'
                    },
                    'csv': {
                        'enabled': True,
                        'solar_path': None,
                        'wind_path': None
                    }
                },
                'preprocessing': {
                    'timezone': 'UTC',
                    'missing_value_strategy': 'interpolate',
                    'outlier_handling': 'flag'
                }
            }
    
    def fetch_entsoe_data(self, start_date: str, end_date: str, 
                         generation_type: str) -> Optional[pd.DataFrame]:
        """
        Fetch data from ENTSO-E API
        
        Parameters:
        -----------
        start_date : str
            Start date in format 'YYYYMMDDHHMM'
        end_date : str
            End date in format 'YYYYMMDDHHMM'
        generation_type : str
            'solar' or 'wind'
        
        Returns:
        --------
        pd.DataFrame or None
            Energy generation data or None if API fails
        """
        if not self.config['data_sources']['entsoe']['enabled']:
            return None
        
        api_key = self.config['data_sources']['entsoe'].get('api_key')
        if not api_key:
            self.logger.warning("ENTSO-E API key not provided. Skipping API fetch.")
            return None
        
        base_url = self.config['data_sources']['entsoe']['base_url']
        
        # ENTSO-E document type codes
        # ActualGenerationOutputPerGenerationUnit: A73
        # ActualWindAndSolarPowerGeneration: A75
        document_type = 'A73' if generation_type == 'solar' else 'A75'
        
        # In-domain code for Germany: 10YDE-VE-------2
        domain = '10YDE-VE-------2'
        
        try:
            url = f"{base_url}"
            params = {
                'securityToken': api_key,
                'documentType': document_type,
                'processType': 'A16',  # Realised
                'in_Domain': domain,
                'periodStart': start_date,
                'periodEnd': end_date
            }
            
            self.logger.info(f"Fetching {generation_type} data from ENTSO-E API...")
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse XML response (simplified - would need proper XML parsing)
            # For now, return None to indicate CSV fallback needed
            self.logger.warning("ENTSO-E XML parsing not fully implemented. Use CSV fallback.")
            return None
            
        except Exception as e:
            self.logger.error(f"ENTSO-E API fetch failed: {e}")
            return None
    
    def load_entsoe_csv(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Load ENTSO-E CSV file with MTU format and 15-minute resolution
        
        Parameters:
        -----------
        file_path : str
            Path to ENTSO-E CSV file
        
        Returns:
        --------
        pd.DataFrame or None
            Dataframe with datetime, solar_generation_mw, wind_generation_mw
        """
        if not Path(file_path).exists():
            self.logger.warning(f"ENTSO-E CSV file not found: {file_path}")
            return None
        
        try:
            self.logger.info(f"Loading ENTSO-E CSV file: {file_path}")
            
            # Read CSV
            df = pd.read_csv(file_path, low_memory=False)
            
            # Check if this is ENTSO-E format
            if 'MTU' not in df.columns:
                self.logger.warning("File does not appear to be ENTSO-E format (no MTU column)")
                return None
            
            # Parse MTU timestamps
            # Format: "01.01.2024 00:00 - 01.01.2024 00:15 (CET/CEST)"
            def parse_mtu(mtu_str):
                """Extract start datetime from MTU string"""
                try:
                    # Extract the start time part (before the dash)
                    start_part = mtu_str.split(' - ')[0].strip()
                    # Parse DD.MM.YYYY HH:MM format
                    dt = pd.to_datetime(start_part, format='%d.%m.%Y %H:%M')
                    return dt
                except:
                    return None
            
            df['datetime'] = df['MTU'].apply(parse_mtu)
            df = df.dropna(subset=['datetime'])
            
            # Handle timezone (CET/CEST)
            # Use Europe/Berlin timezone which automatically handles DST
            try:
                import pytz
                # Localize to Europe/Berlin (CET/CEST)
                # Handle DST transitions: shift_forward for nonexistent times (spring forward)
                # and NaT for ambiguous times (fall back)
                cet_tz = pytz.timezone('Europe/Berlin')
                df['datetime'] = df['datetime'].dt.tz_localize(
                    cet_tz, 
                    ambiguous='NaT',  # Handle fall back (ambiguous hour)
                    nonexistent='shift_forward'  # Handle spring forward (nonexistent hour)
                )
                # Remove any NaT values from ambiguous times (very rare)
                df = df.dropna(subset=['datetime'])
                # Convert to UTC
                df['datetime'] = df['datetime'].dt.tz_convert('UTC')
            except ImportError:
                # Fallback: simple heuristic if pytz not available
                self.logger.warning("pytz not available, using approximate timezone conversion")
                def apply_timezone_offset(dt):
                    """Apply CET/CEST offset to convert to UTC (approximate)"""
                    # For 2024, DST: March 31 02:00 to October 27 03:00
                    if dt.month >= 4 and dt.month <= 9:  # April-September
                        return dt - pd.Timedelta(hours=2)  # CEST = UTC+2
                    elif dt.month == 3 and dt.day >= 31:  # DST starts late March
                        return dt - pd.Timedelta(hours=2)
                    elif dt.month == 10 and dt.day <= 27:  # DST ends late October
                        return dt - pd.Timedelta(hours=2)
                    else:
                        return dt - pd.Timedelta(hours=1)  # CET = UTC+1
                df['datetime'] = df['datetime'].apply(apply_timezone_offset)
            
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Extract solar and wind columns
            solar_col = 'Solar - Actual Aggregated [MW]'
            wind_offshore_col = 'Wind Offshore - Actual Aggregated [MW]'
            wind_onshore_col = 'Wind Onshore - Actual Aggregated [MW]'
            
            if solar_col not in df.columns:
                self.logger.error(f"Solar column not found: {solar_col}")
                return None
            
            if wind_offshore_col not in df.columns or wind_onshore_col not in df.columns:
                self.logger.error(f"Wind columns not found")
                return None
            
            # Convert to numeric, handling "n/e" values
            def convert_to_numeric(series):
                """Convert series to numeric, replacing 'n/e' with NaN"""
                return pd.to_numeric(series.replace('n/e', np.nan), errors='coerce')
            
            df['solar_mw'] = convert_to_numeric(df[solar_col])
            df['wind_offshore_mw'] = convert_to_numeric(df[wind_offshore_col])
            df['wind_onshore_mw'] = convert_to_numeric(df[wind_onshore_col])
            
            # Combine wind offshore + onshore
            df['wind_mw'] = df['wind_offshore_mw'] + df['wind_onshore_mw']
            
            # Aggregate from 15-minute to hourly (mean aggregation)
            self.logger.info("Aggregating 15-minute data to hourly...")
            df['datetime_hour'] = df['datetime'].dt.floor('H')
            
            hourly_df = df.groupby('datetime_hour').agg({
                'solar_mw': 'mean',
                'wind_mw': 'mean'
            }).reset_index()
            hourly_df.rename(columns={'datetime_hour': 'datetime'}, inplace=True)
            
            # Datetime is already timezone-aware (UTC) from groupby, ensure it's UTC
            if hourly_df['datetime'].dt.tz is None:
                hourly_df['datetime'] = hourly_df['datetime'].dt.tz_localize('UTC')
            else:
                hourly_df['datetime'] = hourly_df['datetime'].dt.tz_convert('UTC')
            
            # Rename columns
            hourly_df.rename(columns={
                'solar_mw': 'solar_generation_mw',
                'wind_mw': 'wind_generation_mw'
            }, inplace=True)
            
            # Sort by datetime
            hourly_df = hourly_df.sort_values('datetime').reset_index(drop=True)
            
            self.logger.info(f"Loaded and aggregated {len(hourly_df)} hourly records from {len(df)} 15-minute records")
            self.logger.info(f"Date range: {hourly_df['datetime'].min()} to {hourly_df['datetime'].max()}")
            
            return hourly_df
            
        except Exception as e:
            self.logger.error(f"Error loading ENTSO-E CSV: {e}", exc_info=True)
            return None
    
    def load_csv_data(self, file_path: str, source_type: str) -> Optional[pd.DataFrame]:
        """
        Load energy data from CSV file
        
        Parameters:
        -----------
        file_path : str
            Path to CSV file
        source_type : str
            'solar' or 'wind'
        
        Returns:
        --------
        pd.DataFrame or None
        """
        if not Path(file_path).exists():
            self.logger.warning(f"CSV file not found: {file_path}")
            return None
        
        try:
            self.logger.info(f"Loading {source_type} data from CSV: {file_path}")
            
            # Try different CSV formats
            df = pd.read_csv(file_path, low_memory=False)
            
            # Common column name mappings
            datetime_cols = ['datetime', 'date', 'time', 'timestamp', 'Zeit', 'Datum']
            value_cols = ['generation', 'power', 'production', 'MW', 'GWh', 
                         'solar', 'wind', 'value', 'Wert']
            
            # Find datetime column
            dt_col = None
            for col in datetime_cols:
                if col.lower() in [c.lower() for c in df.columns]:
                    dt_col = col
                    break
            
            if dt_col is None:
                # Try first column
                dt_col = df.columns[0]
                self.logger.warning(f"No recognized datetime column. Using first column: {dt_col}")
            
            # Find value column
            val_col = None
            for col in value_cols:
                if col.lower() in [c.lower() for c in df.columns]:
                    val_col = col
                    break
            
            if val_col is None:
                # Try to find numeric column
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    val_col = numeric_cols[0]
                    self.logger.warning(f"No recognized value column. Using first numeric: {val_col}")
                else:
                    raise ValueError("Could not identify value column")
            
            # Convert datetime
            df[dt_col] = pd.to_datetime(df[dt_col], errors='coerce')
            
            # Create standardized dataframe
            result_df = pd.DataFrame({
                'datetime': df[dt_col],
                f'{source_type}_generation_mw': pd.to_numeric(df[val_col], errors='coerce')
            })
            
            # Remove rows with invalid datetime
            result_df = result_df.dropna(subset=['datetime'])
            
            # Sort by datetime
            result_df = result_df.sort_values('datetime').reset_index(drop=True)
            
            self.logger.info(f"Loaded {len(result_df)} rows from CSV")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error loading CSV: {e}")
            return None
    
    def validate_data(self, df: pd.DataFrame, source_type: str) -> Dict:
        """
        Comprehensive validation of energy data
        
        Parameters:
        -----------
        df : pd.DataFrame
            Energy generation data
        source_type : str
            'solar' or 'wind'
        
        Returns:
        --------
        dict
            Validation report with issues and recommendations
        """
        self.logger.info(f"Validating {source_type} data...")
        
        report = {
            'source_type': source_type,
            'total_rows': len(df),
            'datetime_range': None,
            'missing_hours': [],
            'outliers': [],
            'issues': [],
            'warnings': [],
            'validation_passed': True
        }
        
        if len(df) == 0:
            report['issues'].append("Empty dataset")
            report['validation_passed'] = False
            return report
        
        # Check datetime column
        if 'datetime' not in df.columns:
            report['issues'].append("Missing 'datetime' column")
            report['validation_passed'] = False
            return report
        
        # Check datetime range
        df['datetime'] = pd.to_datetime(df['datetime'])
        min_date = df['datetime'].min()
        max_date = df['datetime'].max()
        report['datetime_range'] = {
            'start': str(min_date),
            'end': str(max_date)
        }
        
        # Check for 2024 data
        if min_date.year != 2024 or max_date.year != 2024:
            report['warnings'].append(f"Data not entirely in 2024: {min_date.year} to {max_date.year}")
        
        # Create expected hourly timeline for 2024
        expected_timeline = pd.date_range(
            start='2024-01-01 00:00:00',
            end='2024-12-31 23:00:00',
            freq='H',
            tz='UTC'
        )
        
        # Check completeness
        df_set = set(df['datetime'].dt.tz_localize('UTC') if df['datetime'].dt.tz is None 
                     else df['datetime'])
        expected_set = set(expected_timeline)
        missing_hours = sorted(expected_set - df_set)
        
        if missing_hours:
            report['missing_hours'] = [str(h) for h in missing_hours[:100]]  # First 100
            missing_pct = len(missing_hours) / len(expected_timeline) * 100
            report['missing_pct'] = missing_pct
            
            if missing_pct > self.validation_config['max_missing_pct']:
                report['issues'].append(
                    f"Too many missing hours: {missing_pct:.2f}% "
                    f"(threshold: {self.validation_config['max_missing_pct']}%)"
                )
                report['validation_passed'] = False
        
        # Check value column
        value_col = f'{source_type}_generation_mw'
        if value_col not in df.columns:
            report['issues'].append(f"Missing value column: {value_col}")
            report['validation_passed'] = False
            return report
        
        # Check for missing values
        missing_values = df[value_col].isna().sum()
        missing_pct = missing_values / len(df) * 100
        if missing_pct > 0:
            report['warnings'].append(f"{missing_pct:.2f}% missing values in generation data")
        
        # Outlier detection
        values = df[value_col].dropna()
        if len(values) > 0:
            z_scores = np.abs((values - values.mean()) / values.std())
            outliers = df[z_scores > self.validation_config['outlier_z_threshold']]
            
            if len(outliers) > 0:
                report['outliers'] = outliers[['datetime', value_col]].head(20).to_dict('records')
                report['warnings'].append(f"Found {len(outliers)} potential outliers (Z > {self.validation_config['outlier_z_threshold']})")
            
            # Range validation
            min_val = values.min()
            max_val = values.max()
            max_capacity = self.validation_config[f'max_{source_type}_capacity_mw']
            
            if max_val > max_capacity:
                report['issues'].append(
                    f"Maximum generation ({max_val:.2f} MW) exceeds expected capacity "
                    f"({max_capacity} MW)"
                )
                report['validation_passed'] = False
            
            if min_val < 0:
                report['issues'].append(f"Negative generation values found: min = {min_val:.2f} MW")
                report['validation_passed'] = False
        
        self.logger.info(f"Validation complete. Passed: {report['validation_passed']}")
        return report
    
    def preprocess_data(self, df: pd.DataFrame, source_type: str) -> pd.DataFrame:
        """
        Preprocess energy data
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw energy data
        source_type : str
            'solar' or 'wind'
        
        Returns:
        --------
        pd.DataFrame
            Preprocessed data
        """
        self.logger.info(f"Preprocessing {source_type} data...")
        
        # Ensure datetime column
        if 'datetime' not in df.columns:
            raise ValueError("Missing datetime column")
        
        df = df.copy()
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Standardize timezone
        if df['datetime'].dt.tz is None:
            df['datetime'] = df['datetime'].dt.tz_localize('UTC')
        else:
            df['datetime'] = df['datetime'].dt.tz_convert('UTC')
        
        # Create complete hourly timeline for 2024
        expected_timeline = pd.date_range(
            start='2024-01-01 00:00:00',
            end='2024-12-31 23:00:00',
            freq='H',
            tz='UTC'
        )
        
        timeline_df = pd.DataFrame({'datetime': expected_timeline})
        
        # Merge with actual data
        value_col = f'{source_type}_generation_mw'
        merge_df = df[['datetime', value_col]].copy()
        result_df = timeline_df.merge(merge_df, on='datetime', how='left')
        
        # Handle missing values
        missing_strategy = self.config['preprocessing'].get('missing_value_strategy', 'interpolate')
        
        if missing_strategy == 'interpolate':
            result_df[value_col] = result_df[value_col].interpolate(method='linear', limit_direction='both')
            # Forward fill remaining NaNs
            result_df[value_col] = result_df[value_col].fillna(method='ffill').fillna(method='bfill')
        elif missing_strategy == 'forward_fill':
            result_df[value_col] = result_df[value_col].fillna(method='ffill').fillna(method='bfill')
        else:
            result_df[value_col] = result_df[value_col].fillna(0)
        
        # Add quality flags
        result_df[f'{source_type}_missing_flag'] = result_df[value_col].isna().astype(int)
        result_df[f'{source_type}_interpolated_flag'] = (result_df[value_col].isna() == False) & \
                                                         (merge_df[value_col].isna()).reindex(
                                                             result_df.index, fill_value=True).astype(int)
        
        # Ensure non-negative
        result_df[value_col] = result_df[value_col].clip(lower=0)
        
        self.logger.info(f"Preprocessing complete. Final dataset: {len(result_df)} rows")
        return result_df
    
    def merge_with_wsi_labels(self, energy_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge energy data with WSI stability labels
        
        Parameters:
        -----------
        energy_df : pd.DataFrame
            Preprocessed energy data
        
        Returns:
        --------
        pd.DataFrame
            Merged dataframe with WSI labels
        """
        self.logger.info("Merging with WSI stability labels...")
        
        if not self.wsi_labels_path.exists():
            self.logger.warning(f"WSI labels not found: {self.wsi_labels_path}. Skipping merge.")
            return energy_df
        
        # Load WSI labels
        wsi_df = pd.read_csv(self.wsi_labels_path)
        wsi_df['datetime'] = pd.to_datetime(wsi_df['datetime'])
        
        # Standardize timezone
        if wsi_df['datetime'].dt.tz is None:
            wsi_df['datetime'] = wsi_df['datetime'].dt.tz_localize('UTC')
        else:
            wsi_df['datetime'] = wsi_df['datetime'].dt.tz_convert('UTC')
        
        # Merge
        merged_df = energy_df.merge(wsi_df, on='datetime', how='left')
        
        self.logger.info(f"Merged with WSI labels. Final dataset: {len(merged_df)} rows")
        return merged_df
    
    def ingest_energy_data(self, source_type: str = 'both') -> Dict:
        """
        Main ingestion pipeline
        
        Parameters:
        -----------
        source_type : str
            'solar', 'wind', or 'both'
        
        Returns:
        --------
        dict
            Ingestion report with validation results
        """
        self.logger.info("="*60)
        self.logger.info("ENERGY DATA INGESTION PIPELINE")
        self.logger.info("="*60)
        
        results = {}
        
        if source_type in ['solar', 'both']:
            results['solar'] = self._ingest_single_source('solar')
        
        if source_type in ['wind', 'both']:
            results['wind'] = self._ingest_single_source('wind')
        
        # Merge solar and wind if both were ingested
        if source_type == 'both' and results['solar']['success'] and results['wind']['success']:
            self.logger.info("Merging solar and wind data...")
            solar_df = pd.read_csv(self.processed_dir / "solar_energy_2024.csv")
            wind_df = pd.read_csv(self.processed_dir / "wind_energy_2024.csv")
            
            # Convert datetime columns
            solar_df['datetime'] = pd.to_datetime(solar_df['datetime'])
            wind_df['datetime'] = pd.to_datetime(wind_df['datetime'])
            
            # Merge on datetime
            merged_df = solar_df.merge(wind_df, on='datetime', how='outer', suffixes=('', '_wind'))
            # Clean up duplicate columns
            merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
            
            # Merge with WSI labels
            merged_df = self.merge_with_wsi_labels(merged_df)
            
            # Save merged dataset
            output_path = self.processed_dir / "energy_production_2024.csv"
            merged_df.to_csv(output_path, index=False)
            self.logger.info(f"Saved merged dataset: {output_path}")
            
            results['merged'] = {
                'success': True,
                'output_path': str(output_path),
                'rows': len(merged_df)
            }
        elif source_type == 'both':
            # If ENTSO-E file was loaded, try to create merged dataset directly
            entsoe_files = list(self.raw_dir.glob("*Generation*Production*Type*.csv"))
            if entsoe_files:
                entsoe_df = self.load_entsoe_csv(str(entsoe_files[0]))
                if entsoe_df is not None and len(entsoe_df) > 0:
                    # Both solar and wind are already in the dataframe
                    merged_df = self.merge_with_wsi_labels(entsoe_df)
                    
                    # Save merged dataset
                    output_path = self.processed_dir / "energy_production_2024.csv"
                    merged_df.to_csv(output_path, index=False)
                    self.logger.info(f"Saved merged dataset from ENTSO-E: {output_path}")
                    
                    results['merged'] = {
                        'success': True,
                        'output_path': str(output_path),
                        'rows': len(merged_df)
                    }
        
        return results
    
    def _ingest_single_source(self, source_type: str) -> Dict:
        """Ingest data for a single source type"""
        self.logger.info(f"\nProcessing {source_type} data...")
        
        result = {
            'source_type': source_type,
            'success': False,
            'validation_report': None,
            'output_path': None
        }
        
        # Try to load data
        df = None
        
        # First, try to find ENTSO-E CSV file in raw directory
        entsoe_files = list(self.raw_dir.glob("*Generation*Production*Type*.csv"))
        if entsoe_files:
            self.logger.info(f"Found ENTSO-E CSV file: {entsoe_files[0]}")
            entsoe_df = self.load_entsoe_csv(str(entsoe_files[0]))
            if entsoe_df is not None and len(entsoe_df) > 0:
                # Extract the requested source type
                if source_type == 'solar':
                    df = entsoe_df[['datetime', 'solar_generation_mw']].copy()
                    df.rename(columns={'solar_generation_mw': 'solar_generation_mw'}, inplace=True)
                elif source_type == 'wind':
                    df = entsoe_df[['datetime', 'wind_generation_mw']].copy()
                    df.rename(columns={'wind_generation_mw': 'wind_generation_mw'}, inplace=True)
        
        # Try CSV first (most reliable) - fallback to config paths
        if df is None or len(df) == 0:
            csv_config = self.config['data_sources']['csv']
            if csv_config['enabled']:
                csv_path = csv_config.get(f'{source_type}_path')
                if csv_path and Path(csv_path).exists():
                    df = self.load_csv_data(csv_path, source_type)
        
        # Try ENTSO-E API if CSV failed
        if df is None or len(df) == 0:
            if self.config['data_sources']['entsoe']['enabled']:
                # Would implement ENTSO-E fetch here
                pass
        
        # If still no data, check raw directory for other CSV files
        if df is None or len(df) == 0:
            raw_files = list(self.raw_dir.glob(f"*{source_type}*.csv"))
            if raw_files:
                self.logger.info(f"Found raw file: {raw_files[0]}")
                df = self.load_csv_data(str(raw_files[0]), source_type)
        
        if df is None or len(df) == 0:
            self.logger.error(f"Could not load {source_type} data from any source")
            result['error'] = "No data source available"
            return result
        
        # Validate
        validation_report = self.validate_data(df, source_type)
        result['validation_report'] = validation_report
        
        if not validation_report['validation_passed']:
            self.logger.error(f"Validation failed for {source_type} data")
            return result
        
        # Preprocess
        processed_df = self.preprocess_data(df, source_type)
        
        # Save processed data
        output_path = self.processed_dir / f"{source_type}_energy_2024.csv"
        processed_df.to_csv(output_path, index=False)
        self.logger.info(f"Saved processed data: {output_path}")
        
        result['success'] = True
        result['output_path'] = str(output_path)
        result['rows'] = len(processed_df)
        
        # Save validation report
        report_path = self.results_dir / f"{source_type}_validation_report.json"
        with open(report_path, 'w') as f:
            json.dump(validation_report, f, indent=2, default=str)
        
        return result


def main():
    """Main execution"""
    ingester = EnergyDataIngester()
    
    # Ingest both solar and wind
    results = ingester.ingest_energy_data(source_type='both')
    
    # Print summary
    print("\n" + "="*60)
    print("INGESTION SUMMARY")
    print("="*60)
    
    for source, result in results.items():
        if isinstance(result, dict) and 'success' in result:
            status = "OK" if result['success'] else "FAILED"
            print(f"{source.upper()}: {status}")
            if result.get('output_path'):
                print(f"  Output: {result['output_path']}")
            if result.get('validation_report'):
                report = result['validation_report']
                print(f"  Rows: {report.get('total_rows', 'N/A')}")
                print(f"  Issues: {len(report.get('issues', []))}")
                print(f"  Warnings: {len(report.get('warnings', []))}")
    
    print("\nIngestion complete. Check logs for details.")


if __name__ == "__main__":
    main()

