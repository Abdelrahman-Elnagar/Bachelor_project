#!/usr/bin/env python3
"""
Script to filter soil temperature data to keep only 2024 records and remove empty files.
"""

import os
import pandas as pd
import glob
from pathlib import Path

def filter_soil_temperature_2024():
    """
    Filter all CSV files in soil_temperature directory to keep only 2024 data.
    Remove files that become empty after filtering.
    """
    
    # Define the soil temperature directory
    soil_temp_dir = Path("D:/Bachelor abroad/soil_temperature")
    
    # Get all CSV files in the directory
    csv_files = list(soil_temp_dir.glob("*.csv"))
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    files_processed = 0
    files_removed = 0
    total_records_before = 0
    total_records_after = 0
    
    for csv_file in csv_files:
        try:
            print(f"Processing {csv_file.name}...")
            
            # Read the CSV file
            df = pd.read_csv(csv_file)
            
            # Count records before filtering
            records_before = len(df)
            total_records_before += records_before
            
            # Filter for 2024 data only
            # The MESS_DATUM column contains datetime strings like "1951-01-01 07:00:00"
            df['MESS_DATUM'] = pd.to_datetime(df['MESS_DATUM'])
            df_2024 = df[df['MESS_DATUM'].dt.year == 2024]
            
            # Count records after filtering
            records_after = len(df_2024)
            total_records_after += records_after
            
            print(f"  Records before: {records_before}, Records after: {records_after}")
            
            if records_after == 0:
                # File becomes empty, remove it
                csv_file.unlink()
                files_removed += 1
                print(f"  Removed empty file: {csv_file.name}")
            else:
                # Save the filtered data back to the file
                df_2024.to_csv(csv_file, index=False)
                files_processed += 1
                print(f"  Updated file: {csv_file.name}")
                
        except Exception as e:
            print(f"Error processing {csv_file.name}: {str(e)}")
            continue
    
    print(f"\nProcessing complete!")
    print(f"Files processed: {files_processed}")
    print(f"Files removed (empty): {files_removed}")
    print(f"Total records before filtering: {total_records_before:,}")
    print(f"Total records after filtering: {total_records_after:,}")
    print(f"Records removed: {total_records_before - total_records_after:,}")

if __name__ == "__main__":
    filter_soil_temperature_2024()
