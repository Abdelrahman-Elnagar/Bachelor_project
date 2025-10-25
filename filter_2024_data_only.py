#!/usr/bin/env python3
"""
Remove all CSV files that don't contain any 2024 data.
Only keep files that have at least one record from 2024.
"""

import os
import pandas as pd
import glob

def has_2024_data(file_path):
    """Check if a CSV file contains any 2024 data."""
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Check if MESS_DATUM column exists
        if 'MESS_DATUM' not in df.columns:
            print(f"⚠ No MESS_DATUM column found in {os.path.basename(file_path)}")
            return False
        
        # Check if any date contains '2024'
        has_2024 = df['MESS_DATUM'].astype(str).str.contains('2024', na=False).any()
        return has_2024
        
    except Exception as e:
        print(f"✗ Error reading {os.path.basename(file_path)}: {str(e)}")
        return False

def filter_2024_files():
    """Remove all files that don't contain 2024 data."""
    input_dir = r"D:\Bachelor abroad\Cloudness_downloads"
    
    # Get all CSV files except the station description file
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    
    # Filter out the station description file
    csv_files = [f for f in csv_files if "N_Stundenwerte_Beschreibung_Stationen" not in f]
    
    if not csv_files:
        print("No CSV files found to process.")
        return
    
    print(f"Found {len(csv_files)} CSV files to check for 2024 data...")
    print("-" * 60)
    
    files_with_2024 = 0
    files_without_2024 = 0
    files_to_remove = []
    
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        
        if has_2024_data(csv_file):
            print(f"✓ {filename} - Contains 2024 data (KEEP)")
            files_with_2024 += 1
        else:
            print(f"✗ {filename} - No 2024 data (REMOVE)")
            files_without_2024 += 1
            files_to_remove.append(csv_file)
    
    print("-" * 60)
    print(f"Analysis complete!")
    print(f"✓ Files with 2024 data: {files_with_2024}")
    print(f"✗ Files without 2024 data: {files_without_2024}")
    
    if files_to_remove:
        print(f"\nRemoving {len(files_to_remove)} files without 2024 data...")
        removed_count = 0
        
        for file_to_remove in files_to_remove:
            try:
                os.remove(file_to_remove)
                print(f"🗑️ Removed: {os.path.basename(file_to_remove)}")
                removed_count += 1
            except Exception as e:
                print(f"✗ Error removing {os.path.basename(file_to_remove)}: {str(e)}")
        
        print(f"\n✅ Successfully removed {removed_count} files")
        print(f"📁 {files_with_2024} files with 2024 data remain")
    else:
        print("\n✅ All files contain 2024 data - no files to remove")

if __name__ == "__main__":
    filter_2024_files()
