#!/usr/bin/env python3
"""
Rename columns to meaningful names across all weather data files

This script renames cryptic column names (like MESS_DATUM, V_N_mean, etc.) 
to more meaningful names (datetime, cloudiness_mean, etc.)
"""

import pandas as pd
import os
import glob
from pathlib import Path
import yaml

class ColumnRenamer:
    def __init__(self):
        """Initialize with column name mappings"""
        
        # Common column mappings (apply to all files)
        self.common_mappings = {
            'MESS_DATUM': 'datetime',
            'STATIONS_ID': 'station_id',
            'STATION_COUNT': 'station_count',
            'STATIONS_CONTRIBUTING': 'contributing_stations',
            'outlier_flag': 'outlier_flag'  # Keep as is
        }
        
        # Quality indicator mappings
        self.quality_mappings = {
            'QN_8': 'quality_indicator',
            'QN_9': 'quality_indicator',
            'QN_3': 'quality_indicator'
        }
        
        # Attribute-specific column mappings
        self.attribute_mappings = {
            'temperature': {
                'TT_TU_mean': 'temperature_mean',
                'TT_TU_min': 'temperature_min',
                'TT_TU_max': 'temperature_max',
                'RF_TU_mean': 'humidity_mean',
                'RF_TU_min': 'humidity_min',
                'RF_TU_max': 'humidity_max'
            },
            'cloudiness': {
                'V_N_mean': 'cloudiness_mean',
                'V_N': 'cloudiness',
                'V_N_I_mean': 'cloudiness_indicator_mean',
                'V_N_I': 'cloudiness_indicator'
            },
            'wind_synop': {
                'FF': 'wind_speed_synop',
                '  FF': 'wind_speed_synop',  # Handle spaces
                'DD': 'wind_direction_synop',
                '  DD': 'wind_direction_synop'
            },
            'wind': {
                'F': 'wind_speed',
                '  F': 'wind_speed',  # Handle spaces
                'D': 'wind_direction',
                '  D': 'wind_direction'
            },
            'precipitation': {
                'R1': 'precipitation_amount',
                '  R1': 'precipitation_amount',  # Handle spaces
                'RS_IND': 'precipitation_indicator',
                'precipitation_indicator': 'precipitation_indicator',  # Already renamed
                'WRTR': 'precipitation_form',
                'precipitation_form': 'precipitation_form'  # Already renamed
            },
            'pressure': {
                'P': 'pressure',
                '   P': 'pressure',  # Handle spaces
                '  P': 'pressure',
                'P0': 'pressure_station_level',
                '  P0': 'pressure_station_level'  # Handle spaces
            },
            'dew_point': {
                'TT': 'temperature_dew_point',
                '  TT': 'temperature_dew_point',  # Handle spaces
                'TD': 'dew_point',
                '  TD': 'dew_point'  # Handle spaces
            },
            'moisture': {
                'ABSF_STD': 'absolute_humidity_std',
                'VP_STD': 'vapor_pressure_std',
                'TF_STD': 'temperature_std',
                'P_STD': 'pressure_std',
                'TT_STD': 'temperature_std',
                'RF_STD': 'humidity_std',
                'TD_STD': 'dew_point_std'
            },
            'extreme_wind': {
                'FX': 'extreme_wind_speed',
                'FX_911': 'extreme_wind_speed_911'  # 911 may be height in m
            },
            'soil_temperature': {
                'EBT': 'soil_temperature',
                'TF': 'soil_temperature_depth'  # May vary by depth
            },
            'sun': {
                'SD_SO': 'sunshine_duration',
                'SO': 'sunshine_duration_alt'
            },
            'visibility': {
                'V_VV': 'visibility',
                'VV': 'visibility_alt'
            },
            'weather_phenomena': {
                # These may vary, common ones:
                'WW': 'weather_phenomena_code',
                'WW_TEXT': 'weather_phenomena_text'
            }
        }
    
    def get_rename_dict(self, df_columns, attribute_name=None):
        """Get rename dictionary for given columns"""
        rename_dict = {}
        
        # Apply common mappings
        for old_name, new_name in self.common_mappings.items():
            if old_name in df_columns:
                rename_dict[old_name] = new_name
        
        # Apply quality indicator mappings
        for old_name, new_name in self.quality_mappings.items():
            if old_name in df_columns:
                rename_dict[old_name] = new_name
        
        # Apply attribute-specific mappings if attribute is known
        if attribute_name and attribute_name in self.attribute_mappings:
            for old_name, new_name in self.attribute_mappings[attribute_name].items():
                if old_name in df_columns:
                    rename_dict[old_name] = new_name
        
        # Handle stations column (already processed, might be lowercase)
        if 'stations' in df_columns and 'stations' not in rename_dict.values():
            rename_dict['stations'] = 'contributing_stations'
        
        return rename_dict
    
    def rename_file(self, file_path, attribute_name=None):
        """Rename columns in a single file"""
        try:
            # Read CSV
            df = pd.read_csv(file_path, low_memory=False)
            
            # Get rename dictionary
            rename_dict = self.get_rename_dict(df.columns, attribute_name)
            
            if not rename_dict:
                return False  # No renaming needed
            
            # Rename columns
            df.rename(columns=rename_dict, inplace=True)
            
            # Save file
            df.to_csv(file_path, index=False)
            
            return True, list(rename_dict.keys()), list(rename_dict.values())
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return False, None, None
    
    def process_folder(self, folder_path, attribute_name=None):
        """Process all CSV files in a folder"""
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        renamed_count = 0
        
        for file_path in csv_files:
            # Skip summary files
            if '_SUMMARY' in os.path.basename(file_path).upper() or \
               '_AGGREGATION' in os.path.basename(file_path).upper():
                continue
            
            result = self.rename_file(file_path, attribute_name)
            if result and result[0]:
                renamed_count += 1
                print(f"  Renamed columns in: {os.path.basename(file_path)}")
        
        return renamed_count


def main():
    """Main function to rename columns across all files"""
    base_path = Path("Data")
    renamer = ColumnRenamer()
    
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
    
    print("="*60)
    print("RENAMING COLUMNS TO MEANINGFUL NAMES")
    print("="*60)
    
    total_renamed = 0
    
    for attr_name, bundesland_folder, germany_folder in attributes:
        print(f"\nProcessing {attr_name}...")
        
        # Process Bundesland folder
        bundesland_path = base_path / bundesland_folder
        if bundesland_path.exists():
            count = renamer.process_folder(bundesland_path, attr_name)
            total_renamed += count
            print(f"  Renamed columns in {count} files in {bundesland_folder}")
        else:
            print(f"  Folder not found: {bundesland_path}")
        
        # Process Germany folder
        germany_path = base_path / germany_folder
        if germany_path.exists():
            count = renamer.process_folder(germany_path, attr_name)
            total_renamed += count
            print(f"  Renamed columns in {count} files in {germany_folder}")
        else:
            print(f"  Folder not found: {germany_path}")
    
    print(f"\n{'='*60}")
    print(f"Column renaming complete!")
    print(f"Total files processed: {total_renamed}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

