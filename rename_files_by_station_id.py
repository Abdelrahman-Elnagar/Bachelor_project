#!/usr/bin/env python3
"""
Rename CSV files based on the STATIONS_ID found in each file.
Files will be renamed from: produkt_n_stunde_YYYYMMDD_YYYYMMDD_XXXXX.csv
To: station_XXXXX_YYYYMMDD_YYYYMMDD.csv
"""

import os
import pandas as pd
import glob
import re

def get_station_id_from_file(file_path):
    """Extract STATIONS_ID from the first data row of a CSV file."""
    try:
        # Read only the first few rows to get the STATIONS_ID
        df = pd.read_csv(file_path, nrows=5)
        if 'STATIONS_ID' in df.columns and len(df) > 0:
            # Get the first non-null STATIONS_ID value
            station_id = df['STATIONS_ID'].iloc[0]
            return str(station_id).strip()
        return None
    except Exception as e:
        print(f"Error reading {os.path.basename(file_path)}: {str(e)}")
        return None

def extract_date_range_from_filename(filename):
    """Extract date range from the original filename."""
    # Pattern: produkt_n_stunde_YYYYMMDD_YYYYMMDD_XXXXX.csv
    pattern = r'produkt_n_stunde_(\d{8})_(\d{8})_\d+\.csv'
    match = re.match(pattern, filename)
    if match:
        start_date = match.group(1)
        end_date = match.group(2)
        return start_date, end_date
    return None, None

def rename_files_by_station_id():
    """Rename all CSV files based on their STATIONS_ID."""
    input_dir = r"D:\Bachelor abroad\Cloudness_downloads"
    
    # Get all CSV files except the station description file
    csv_files = glob.glob(os.path.join(input_dir, "produkt_n_stunde_*.csv"))
    
    if not csv_files:
        print("No CSV files found to rename.")
        return
    
    print(f"Found {len(csv_files)} CSV files to process...")
    print("-" * 60)
    
    successful_renames = 0
    failed_renames = 0
    skipped_files = 0
    
    for csv_file in csv_files:
        try:
            # Get the current filename
            current_filename = os.path.basename(csv_file)
            
            # Skip the station description file
            if "N_Stundenwerte_Beschreibung_Stationen" in current_filename:
                print(f"⚠ Skipping station description file: {current_filename}")
                skipped_files += 1
                continue
            
            # Extract STATIONS_ID from the file
            station_id = get_station_id_from_file(csv_file)
            if not station_id:
                print(f"✗ Could not extract STATIONS_ID from: {current_filename}")
                failed_renames += 1
                continue
            
            # Extract date range from filename
            start_date, end_date = extract_date_range_from_filename(current_filename)
            if not start_date or not end_date:
                print(f"✗ Could not extract date range from: {current_filename}")
                failed_renames += 1
                continue
            
            # Create new filename: station_XXXXX_YYYYMMDD_YYYYMMDD.csv
            new_filename = f"station_{station_id}_{start_date}_{end_date}.csv"
            new_file_path = os.path.join(input_dir, new_filename)
            
            # Check if target file already exists
            if os.path.exists(new_file_path):
                print(f"⚠ Target file already exists, skipping: {new_filename}")
                skipped_files += 1
                continue
            
            # Rename the file
            os.rename(csv_file, new_file_path)
            print(f"✓ Renamed: {current_filename} → {new_filename}")
            successful_renames += 1
            
        except Exception as e:
            print(f"✗ Error renaming {current_filename}: {str(e)}")
            failed_renames += 1
    
    print("-" * 60)
    print(f"Rename operation complete!")
    print(f"✓ Successfully renamed: {successful_renames} files")
    print(f"⚠ Skipped: {skipped_files} files")
    print(f"✗ Failed renames: {failed_renames} files")
    
    if successful_renames > 0:
        print(f"\nFiles renamed in: {input_dir}")
        print("\nNew naming convention: station_STATIONID_STARTDATE_ENDDATE.csv")
        print("Example: station_13777_20080601_20120109.csv")

if __name__ == "__main__":
    rename_files_by_station_id()
