#!/usr/bin/env python3
"""
Convert all text files in Cloudness_downloads folder to CSV format.
Handles both semicolon-separated data files and the station description file.
"""

import os
import pandas as pd
import glob
from pathlib import Path

def convert_semicolon_to_csv(input_file, output_file):
    """Convert semicolon-separated file to CSV format."""
    try:
        # Read the semicolon-separated file
        df = pd.read_csv(input_file, sep=';', encoding='utf-8')
        
        # Remove the 'eor' column if it exists (seems to be a terminator column)
        if 'eor' in df.columns:
            df = df.drop('eor', axis=1)
        
        # Save as CSV
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"✓ Converted: {os.path.basename(input_file)} -> {os.path.basename(output_file)}")
        return True
    except Exception as e:
        print(f"✗ Error converting {os.path.basename(input_file)}: {str(e)}")
        return False

def convert_station_description_to_csv(input_file, output_file):
    """Convert the station description file to CSV format."""
    try:
        # Read the file line by line to handle the special format
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Skip the header separator line (line 2)
        data_lines = []
        for i, line in enumerate(lines):
            if i == 0:  # Header line
                header = line.strip()
            elif i == 1:  # Skip separator line
                continue
            elif line.strip():  # Data lines
                data_lines.append(line.strip())
        
        # Parse the data
        records = []
        for line in data_lines:
            # Split by whitespace but preserve the structure
            parts = line.split()
            if len(parts) >= 8:  # Ensure we have enough columns
                record = {
                    'Stations_id': parts[0],
                    'von_datum': parts[1],
                    'bis_datum': parts[2],
                    'Stationshoehe': parts[3],
                    'geoBreite': parts[4],
                    'geoLaenge': parts[5],
                    'Stationsname': ' '.join(parts[6:-2]) if len(parts) > 8 else parts[6],
                    'Bundesland': parts[-2] if len(parts) > 7 else '',
                    'Abgabe': parts[-1] if len(parts) > 8 else ''
                }
                records.append(record)
        
        # Create DataFrame and save as CSV
        df = pd.DataFrame(records)
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"✓ Converted: {os.path.basename(input_file)} -> {os.path.basename(output_file)}")
        return True
    except Exception as e:
        print(f"✗ Error converting {os.path.basename(input_file)}: {str(e)}")
        return False

def main():
    """Main conversion function."""
    # Set the input and output directories
    input_dir = r"D:\Bachelor abroad\Cloudness_downloads"
    output_dir = r"D:\Bachelor abroad\Cloudness_downloads"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all .txt files in the directory
    txt_files = glob.glob(os.path.join(input_dir, "*.txt"))
    
    if not txt_files:
        print("No .txt files found in the directory.")
        return
    
    print(f"Found {len(txt_files)} .txt files to convert...")
    print("-" * 50)
    
    successful_conversions = 0
    failed_conversions = 0
    
    for txt_file in txt_files:
        # Create output filename
        base_name = os.path.splitext(os.path.basename(txt_file))[0]
        csv_file = os.path.join(output_dir, f"{base_name}.csv")
        
        # Skip if CSV already exists
        if os.path.exists(csv_file):
            print(f"⚠ Skipping {os.path.basename(txt_file)} (CSV already exists)")
            continue
        
        # Determine conversion method based on filename
        if "N_Stundenwerte_Beschreibung_Stationen" in txt_file:
            # Special handling for station description file
            if convert_station_description_to_csv(txt_file, csv_file):
                successful_conversions += 1
            else:
                failed_conversions += 1
        else:
            # Standard semicolon-separated data files
            if convert_semicolon_to_csv(txt_file, csv_file):
                successful_conversions += 1
            else:
                failed_conversions += 1
    
    print("-" * 50)
    print(f"Conversion complete!")
    print(f"✓ Successfully converted: {successful_conversions} files")
    print(f"✗ Failed conversions: {failed_conversions} files")
    
    if successful_conversions > 0:
        print(f"\nCSV files saved in: {output_dir}")

if __name__ == "__main__":
    main()
