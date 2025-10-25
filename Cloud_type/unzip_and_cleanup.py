#!/usr/bin/env python3
"""
Script to unzip all zip files in the current directory and delete the zip files after extraction.
"""

import os
import zipfile
import glob
from pathlib import Path

def unzip_and_delete_zip_files():
    """Unzip all zip files in the current directory and delete them after extraction."""
    
    # Get current directory
    current_dir = Path.cwd()
    
    # Find all zip files in the current directory
    zip_files = list(current_dir.glob("*.zip"))
    
    if not zip_files:
        print("No zip files found in the current directory.")
        return
    
    print(f"Found {len(zip_files)} zip files to process...")
    
    successful_extractions = 0
    failed_extractions = 0
    
    for i, zip_file in enumerate(zip_files, 1):
        print(f"\nProgress: {i}/{len(zip_files)}")
        print(f"Processing: {zip_file.name}")
        
        try:
            # Extract the zip file
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                # Get list of files in the zip
                file_list = zip_ref.namelist()
                print(f"  Extracting {len(file_list)} files...")
                
                # Extract all files
                zip_ref.extractall(current_dir)
                
                # Delete the zip file after successful extraction
                zip_file.unlink()
                print(f"  ✓ Successfully extracted and deleted {zip_file.name}")
                successful_extractions += 1
                
        except zipfile.BadZipFile:
            print(f"  ✗ Error: {zip_file.name} is not a valid zip file")
            failed_extractions += 1
        except Exception as e:
            print(f"  ✗ Error processing {zip_file.name}: {str(e)}")
            failed_extractions += 1
    
    print(f"\n{'='*50}")
    print(f"Extraction complete!")
    print(f"Successfully processed: {successful_extractions}")
    print(f"Failed extractions: {failed_extractions}")
    
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
    unzip_and_delete_zip_files()

