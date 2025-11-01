#!/usr/bin/env python3
"""
Filter all CSV files to contain only 2024 data.
Remove all historical data (pre-2024) from each file.
"""

import os
import pandas as pd
import glob

def filter_to_2024_only(file_path):
    """Filter a CSV file to contain only 2024 data."""
    try:
        filename = os.path.basename(file_path)
        print(f"Processing: {filename}")
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Check if MESS_DATUM column exists
        if 'MESS_DATUM' not in df.columns:
            print(f"⚠ No MESS_DATUM column found in {filename}")
            return False
        
        # Convert MESS_DATUM to datetime for filtering
        df['MESS_DATUM'] = pd.to_datetime(df['MESS_DATUM'])
        
        # Filter to only 2024 data
        df_2024 = df[df['MESS_DATUM'].dt.year == 2024]
        
        # Check if any 2024 data exists
        if len(df_2024) == 0:
            print(f"⚠ No 2024 data found in {filename} - removing file")
            os.remove(file_path)
            return False
        
        # Convert back to string format for consistency
        df_2024['MESS_DATUM'] = df_2024['MESS_DATUM'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Save the filtered data back to the file
        df_2024.to_csv(file_path, index=False)
        
        print(f"✓ Filtered {filename}: {len(df)} → {len(df_2024)} records (2024 only)")
        return True
        
    except Exception as e:
        print(f"✗ Error processing {os.path.basename(file_path)}: {str(e)}")
        return False

def filter_all_files_to_2024():
    """Filter all CSV files to contain only 2024 data."""
    input_dir = r"D:\Bachelor abroad\Cloudness_downloads"
    
    # Get all CSV files except the station description file
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    
    # Filter out the station description file
    csv_files = [f for f in csv_files if "N_Stundenwerte_Beschreibung_Stationen" not in f]
    
    if not csv_files:
        print("No CSV files found to process.")
        return
    
    print(f"Found {len(csv_files)} CSV files to filter to 2024 data only...")
    print("-" * 80)
    
    successful_filters = 0
    failed_filters = 0
    files_removed = 0
    
    for csv_file in csv_files:
        if filter_to_2024_only(csv_file):
            successful_filters += 1
        else:
            failed_filters += 1
            if not os.path.exists(csv_file):  # File was removed
                files_removed += 1
                failed_filters -= 1  # Don't count removed files as failures
    
    print("-" * 80)
    print(f"2024-only filtering complete!")
    print(f"✓ Successfully filtered: {successful_filters} files")
    print(f"🗑️ Files removed (no 2024 data): {files_removed} files")
    print(f"✗ Failed filters: {failed_filters} files")
    
    if successful_filters > 0:
        print(f"\nAll remaining files now contain ONLY 2024 data")
        print(f"Historical data (pre-2024) has been removed from all files")

if __name__ == "__main__":
    filter_all_files_to_2024()
