#!/usr/bin/env python3
"""
Final fix for column names - handles remaining issues with spaces and duplicates
"""

import pandas as pd
import os
import glob
from pathlib import Path

def fix_column_names_in_file(file_path, attribute_name=None):
    """Fix column names in a single file"""
    try:
        df = pd.read_csv(file_path, low_memory=False)
        renamed = False
        
        # Fix columns with leading spaces
        column_mappings = {
            '   P': 'pressure',
            '  P': 'pressure',
            'P': 'pressure',
            '  P0': 'pressure_station_level',
            'P0': 'pressure_station_level',
            '   F': 'wind_speed',
            '  F': 'wind_speed',
            'F': 'wind_speed',
            '   D': 'wind_direction',
            '  D': 'wind_direction',
            'D': 'wind_direction',
            '  TT': 'temperature_dew_point',
            'TT': 'temperature_dew_point',
            '  TD': 'dew_point',
            'TD': 'dew_point',
        }
        
        # Rename columns with spaces
        rename_dict = {}
        for old_col in df.columns:
            if old_col in column_mappings:
                rename_dict[old_col] = column_mappings[old_col]
                renamed = True
        
        if rename_dict:
            df.rename(columns=rename_dict, inplace=True)
        
        # Remove duplicate station_id columns (keep first one)
        if 'station_id.1' in df.columns:
            df.drop(columns=['station_id.1'], inplace=True)
            renamed = True
        
        if renamed:
            df.to_csv(file_path, index=False)
            return True
        return False
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False

def fix_all_files():
    """Fix all CSV files"""
    base_path = Path("Data")
    
    attributes_folders = [
        ('pressure', 'Bundesland_aggregation/pressure_by_bundesland', 'Germany_aggregation/pressure_germany'),
        ('wind', 'Bundesland_aggregation/wind_by_bundesland', 'Germany_aggregation/wind_germany'),
        ('dew_point', 'Bundesland_aggregation/dew_point_by_bundesland', 'Germany_aggregation/dew_point_germany_aggregated'),
        ('wind_synop', 'Bundesland_aggregation/wind_synop_by_bundesland', 'Germany_aggregation/wind_synop_germany'),
        ('precipitation', 'Bundesland_aggregation/precipitation_by_bundesland', 'Germany_aggregation/precipitation_germany'),
    ]
    
    print("Fixing column names with spaces and duplicates...")
    
    for attr_name, bundesland_folder, germany_folder in attributes_folders:
        # Fix Bundesland files
        bundesland_path = base_path / bundesland_folder
        if bundesland_path.exists():
            csv_files = list(bundesland_path.glob("*.csv"))
            fixed_count = 0
            for file_path in csv_files:
                if '_SUMMARY' not in file_path.name.upper():
                    if fix_column_names_in_file(file_path, attr_name):
                        fixed_count += 1
            if fixed_count > 0:
                print(f"  Fixed {fixed_count} files in {bundesland_folder}")
        
        # Fix Germany files
        germany_path = base_path / germany_folder
        if germany_path.exists():
            csv_files = list(germany_path.glob("*.csv"))
            fixed_count = 0
            for file_path in csv_files:
                if '_SUMMARY' not in file_path.name.upper():
                    if fix_column_names_in_file(file_path, attr_name):
                        fixed_count += 1
            if fixed_count > 0:
                print(f"  Fixed {fixed_count} files in {germany_folder}")
    
    print("Column name fixes complete!")

if __name__ == "__main__":
    fix_all_files()

