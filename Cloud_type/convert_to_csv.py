#!/usr/bin/env python3
"""
Script to convert all text files to CSV format.
This script reads all .txt files in the current directory and converts them to .csv files.
"""

import os
import csv
from pathlib import Path
import glob

def convert_txt_to_csv():
    """Convert all .txt files in the current directory to .csv files."""
    
    # Get current directory
    current_dir = Path.cwd()
    
    # Find all txt files in the current directory
    txt_files = list(current_dir.glob("*.txt"))
    
    if not txt_files:
        print("No .txt files found in the current directory.")
        return
    
    print(f"Found {len(txt_files)} .txt files to convert...")
    
    successful_conversions = 0
    failed_conversions = 0
    
    for i, txt_file in enumerate(txt_files, 1):
        print(f"\nProgress: {i}/{len(txt_files)}")
        print(f"Converting: {txt_file.name}")
        
        try:
            # Create output filename
            csv_file = txt_file.with_suffix('.csv')
            
            # Skip if CSV file already exists
            if csv_file.exists():
                print(f"  CSV file already exists, skipping...")
                continue
            
            # Read the txt file and convert to CSV
            with open(txt_file, 'r', encoding='utf-8') as infile:
                # Read all lines
                lines = infile.readlines()
                
                if not lines:
                    print(f"  Warning: {txt_file.name} is empty, skipping...")
                    continue
                
                # Process the data
                with open(csv_file, 'w', newline='', encoding='utf-8') as outfile:
                    csv_writer = csv.writer(outfile)
                    
                    for line_num, line in enumerate(lines, 1):
                        line = line.strip()
                        if not line:
                            continue
                            
                        # Split by semicolon and clean up the data
                        fields = [field.strip() for field in line.split(';')]
                        
                        # Remove 'eor' column (last column)
                        if fields and fields[-1] == 'eor':
                            fields = fields[:-1]
                        
                        # Write the row to CSV
                        csv_writer.writerow(fields)
            
            # Remove the original txt file after successful conversion
            txt_file.unlink()
            print(f"  ✓ Successfully converted {txt_file.name} to {csv_file.name}")
            successful_conversions += 1
            
        except Exception as e:
            print(f"  ✗ Error converting {txt_file.name}: {str(e)}")
            failed_conversions += 1
    
    print(f"\n{'='*50}")
    print(f"Conversion complete!")
    print(f"Successfully converted: {successful_conversions}")
    print(f"Failed conversions: {failed_conversions}")
    
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
    convert_txt_to_csv()
