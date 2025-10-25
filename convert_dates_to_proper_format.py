#!/usr/bin/env python3
"""
Convert date format in all CSV files from YYYYMMDDHH to YYYY-MM-DD HH:MM:SS format.
This script processes all CSV files in the Cloudness_downloads folder.
"""

import os
import pandas as pd
import glob
from datetime import datetime

def convert_date_format(date_str):
    """Convert date from YYYYMMDDHH format to YYYY-MM-DD HH:MM:SS format."""
    try:
        # Parse the original format: YYYYMMDDHH
        if len(str(date_str)) == 10:
            year = str(date_str)[:4]
            month = str(date_str)[4:6]
            day = str(date_str)[6:8]
            hour = str(date_str)[8:10]
            
            # Create proper datetime format
            formatted_date = f"{year}-{month}-{day} {hour}:00:00"
            return formatted_date
        else:
            return date_str  # Return as-is if format doesn't match
    except Exception as e:
        print(f"Error converting date {date_str}: {str(e)}")
        return date_str

def process_csv_file(file_path):
    """Process a single CSV file to convert date format."""
    try:
        print(f"Processing: {os.path.basename(file_path)}")
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Check if MESS_DATUM column exists
        if 'MESS_DATUM' not in df.columns:
            print(f"⚠ No MESS_DATUM column found in {os.path.basename(file_path)}")
            return False
        
        # Convert the date format
        df['MESS_DATUM'] = df['MESS_DATUM'].apply(convert_date_format)
        
        # Save the updated CSV file
        df.to_csv(file_path, index=False)
        
        print(f"✓ Converted dates in: {os.path.basename(file_path)}")
        return True
        
    except Exception as e:
        print(f"✗ Error processing {os.path.basename(file_path)}: {str(e)}")
        return False

def convert_all_dates():
    """Convert date format in all CSV files."""
    input_dir = r"D:\Bachelor abroad\Cloudness_downloads"
    
    # Get all CSV files except the station description file
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    
    # Filter out the station description file
    csv_files = [f for f in csv_files if "N_Stundenwerte_Beschreibung_Stationen" not in f]
    
    if not csv_files:
        print("No CSV files found to process.")
        return
    
    print(f"Found {len(csv_files)} CSV files to process...")
    print("-" * 60)
    
    successful_conversions = 0
    failed_conversions = 0
    
    for csv_file in csv_files:
        if process_csv_file(csv_file):
            successful_conversions += 1
        else:
            failed_conversions += 1
    
    print("-" * 60)
    print(f"Date conversion complete!")
    print(f"✓ Successfully converted: {successful_conversions} files")
    print(f"✗ Failed conversions: {failed_conversions} files")
    
    if successful_conversions > 0:
        print(f"\nDate format changed from: YYYYMMDDHH (e.g., 2008060106)")
        print(f"Date format changed to: YYYY-MM-DD HH:MM:SS (e.g., 2008-06-01 06:00:00)")

if __name__ == "__main__":
    convert_all_dates()
