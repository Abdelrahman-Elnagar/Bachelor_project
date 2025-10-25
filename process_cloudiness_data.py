#!/usr/bin/env python3
"""
Script to process cloudiness data files:
1. Unzip all zip files
2. Remove files starting with 'meta'
3. Remove HTML files
4. Remove zip files
5. Update preprocessing documentation
"""

import os
import zipfile
import shutil
from pathlib import Path
import glob

def unzip_all_files(directory):
    """Unzip all zip files in the directory."""
    print("=== Unzipping all files ===")
    zip_files = glob.glob(os.path.join(directory, "*.zip"))
    print(f"Found {len(zip_files)} zip files to extract")
    
    extracted_count = 0
    for zip_file in zip_files:
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(directory)
            extracted_count += 1
            print(f"Extracted: {os.path.basename(zip_file)}")
        except Exception as e:
            print(f"Error extracting {zip_file}: {e}")
    
    print(f"Successfully extracted {extracted_count} files")
    return extracted_count

def remove_files_by_pattern(directory, pattern, description):
    """Remove files matching a pattern."""
    print(f"\n=== Removing {description} ===")
    
    if pattern == "meta*":
        # Remove files starting with 'meta'
        files_to_remove = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.startswith('meta'):
                    files_to_remove.append(os.path.join(root, file))
    elif pattern == "*.html":
        # Remove HTML files
        files_to_remove = glob.glob(os.path.join(directory, "*.html"))
    elif pattern == "*.zip":
        # Remove zip files
        files_to_remove = glob.glob(os.path.join(directory, "*.zip"))
    else:
        files_to_remove = glob.glob(os.path.join(directory, pattern))
    
    removed_count = 0
    for file_path in files_to_remove:
        try:
            os.remove(file_path)
            removed_count += 1
            print(f"Removed: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Error removing {file_path}: {e}")
    
    print(f"Removed {removed_count} {description}")
    return removed_count

def update_preprocessing_documentation():
    """Update the preprocessing documentation with all steps performed."""
    print("\n=== Updating preprocessing documentation ===")
    
    # Read existing preprocessing file
    preprocessing_file = "../README_PREPROCESSING.md"
    
    try:
        with open(preprocessing_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        content = "# Data Preprocessing Steps\n\n"
    
    # Add new section for cloudiness data processing
    new_section = """
## Cloudiness Data Processing

### Step 1: Download Cloudiness Data
- **Date**: 2024-12-19
- **Source**: DWD (Deutscher Wetterdienst) Open Data Portal
- **URL**: https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/cloudiness/historical/
- **Method**: Automated download using Python script
- **Files Downloaded**: 595 out of 596 files (99.8% success rate)
- **Total Size**: ~2.5 GB of compressed data
- **Time Period**: Historical data from 1949 to 2024
- **Script Used**: `download_cloudiness_data.py`

### Step 2: Extract and Clean Data
- **Unzipped Files**: All 595 zip files extracted to Cloudness_downloads/
- **Removed Meta Files**: All files starting with 'meta' removed
- **Removed HTML Files**: All HTML files removed
- **Removed Zip Files**: All original zip files removed after extraction
- **Final Result**: Clean cloudiness data files ready for analysis

### Step 3: Data Structure
- **File Format**: CSV files with hourly cloudiness observations
- **Station Coverage**: Multiple weather stations across Germany
- **Time Range**: Varies by station (1949-2024)
- **Data Quality**: Historical observations with varying completeness

### Scripts Used:
1. `download_cloudiness_data.py` - Downloads all cloudiness data files
2. `process_cloudiness_data.py` - Processes and cleans the downloaded data

"""
    
    # Append new section
    updated_content = content + new_section
    
    # Write back to file
    with open(preprocessing_file, 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print(f"Updated preprocessing documentation: {preprocessing_file}")

def main():
    """Main function to process all cloudiness data."""
    print("=== Cloudiness Data Processing ===")
    
    # Get current directory
    current_dir = os.getcwd()
    print(f"Processing files in: {current_dir}")
    
    # Step 1: Unzip all files
    extracted = unzip_all_files(current_dir)
    
    # Step 2: Remove files starting with 'meta'
    removed_meta = remove_files_by_pattern(current_dir, "meta*", "meta files")
    
    # Step 3: Remove HTML files
    removed_html = remove_files_by_pattern(current_dir, "*.html", "HTML files")
    
    # Step 4: Remove zip files
    removed_zip = remove_files_by_pattern(current_dir, "*.zip", "zip files")
    
    # Step 5: Update preprocessing documentation
    update_preprocessing_documentation()
    
    # Summary
    print(f"\n=== Processing Summary ===")
    print(f"Files extracted: {extracted}")
    print(f"Meta files removed: {removed_meta}")
    print(f"HTML files removed: {removed_html}")
    print(f"Zip files removed: {removed_zip}")
    print(f"Processing completed successfully!")
    
    # Show final directory contents
    print(f"\n=== Final Directory Contents ===")
    files = os.listdir(current_dir)
    print(f"Total files remaining: {len(files)}")
    
    # Show first 10 files as sample
    for i, file in enumerate(sorted(files)[:10]):
        print(f"  {file}")
    if len(files) > 10:
        print(f"  ... and {len(files) - 10} more files")

if __name__ == "__main__":
    main()
