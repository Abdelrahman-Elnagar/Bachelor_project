#!/usr/bin/env python3
"""
Script to rename all CSV files with their station ID.
This script reads each CSV file, extracts the station ID from the first data row,
and renames the file to include the station ID.
"""

import os
import csv
from pathlib import Path

def rename_files_with_station_id():
    """Rename all CSV files to include their station ID."""
    
    # Get current directory
    current_dir = Path.cwd()
    
    # Find all CSV files in the current directory
    csv_files = list(current_dir.glob("*.csv"))
    
    if not csv_files:
        print("No .csv files found in the current directory.")
        return
    
    print(f"Found {len(csv_files)} .csv files to rename...")
    
    successful_renames = 0
    failed_renames = 0
    
    for i, csv_file in enumerate(csv_files, 1):
        print(f"\nProgress: {i}/{len(csv_files)}")
        print(f"Processing: {csv_file.name}")
        
        try:
            # Read the CSV file to get the station ID
            with open(csv_file, 'r', encoding='utf-8') as file:
                csv_reader = csv.reader(file)
                
                # Skip header row
                next(csv_reader)
                
                # Get first data row
                first_row = next(csv_reader)
                
                if not first_row:
                    print(f"  Warning: {csv_file.name} has no data rows, skipping...")
                    continue
                
                # Extract station ID from first column
                station_id = first_row[0].strip()
                
                if not station_id:
                    print(f"  Warning: No station ID found in {csv_file.name}, skipping...")
                    continue
                
                # Create new filename with station ID
                # Extract the base name without extension
                base_name = csv_file.stem
                
                # Create new filename: station_id_original_name.csv
                new_filename = f"{station_id}_{base_name}.csv"
                new_file_path = csv_file.parent / new_filename
                
                # Check if new filename already exists
                if new_file_path.exists():
                    print(f"  Warning: {new_filename} already exists, skipping...")
                    continue
                
                # Rename the file
                csv_file.rename(new_file_path)
                print(f"  ✓ Successfully renamed to {new_filename}")
                successful_renames += 1
                
        except Exception as e:
            print(f"  ✗ Error processing {csv_file.name}: {str(e)}")
            failed_renames += 1
    
    print(f"\n{'='*50}")
    print(f"Rename complete!")
    print(f"Successfully renamed: {successful_renames}")
    print(f"Failed renames: {failed_renames}")
    
    # Show remaining files in directory
    remaining_files = list(current_dir.glob("*"))
    print(f"\nRemaining files in directory: {len(remaining_files)}")
    
    # Show file types
    file_types = {}
    for file in remaining_files:
        if file.is_file():
            ext = file.suffix.lower()
            file_types[ext] = file_types.get(ext, 0) + 1
    
    print("\nFile types:")
    for ext, count in sorted(file_types.items()):
        print(f"  {ext or 'no extension'}: {count} files")

if __name__ == "__main__":
    rename_files_with_station_id()

