#!/usr/bin/env python3
"""
Script to remove all metadata files starting with 'Metadaten' from the cloudiness data directory.
This will keep only the actual data files (produkt_n_stunde files).
"""

import os
import glob
from pathlib import Path

def remove_metadata_files(directory):
    """Remove all files starting with 'Metadaten' from the directory."""
    print("=== Removing Metadata Files ===")
    
    # Find all files starting with 'Metadaten
    metadata_files = glob.glob(os.path.join(directory, "Metadaten*"))
    
    print(f"Found {len(metadata_files)} metadata files to remove")
    
    removed_count = 0
    for file_path in metadata_files:
        try:
            os.remove(file_path)
            removed_count += 1
            print(f"Removed: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Error removing {file_path}: {e}")
    
    print(f"Successfully removed {removed_count} metadata files")
    return removed_count

def show_remaining_files(directory):
    """Show the remaining files after cleanup."""
    print(f"\n=== Remaining Files in {directory} ===")
    
    all_files = os.listdir(directory)
    data_files = [f for f in all_files if f.startswith('produkt_n_stunde')]
    other_files = [f for f in all_files if not f.startswith('produkt_n_stunde')]
    
    print(f"Total files remaining: {len(all_files)}")
    print(f"Data files (produkt_n_stunde): {len(data_files)}")
    print(f"Other files: {len(other_files)}")
    
    if other_files:
        print(f"\nOther files found:")
        for file in sorted(other_files)[:10]:  # Show first 10
            print(f"  {file}")
        if len(other_files) > 10:
            print(f"  ... and {len(other_files) - 10} more files")
    
    print(f"\nData files (first 10):")
    for file in sorted(data_files)[:10]:
        print(f"  {file}")
    if len(data_files) > 10:
        print(f"  ... and {len(data_files) - 10} more data files")

def main():
    """Main function to remove metadata files."""
    print("=== Cloudiness Data Cleanup ===")
    
    # Get current directory
    current_dir = os.getcwd()
    print(f"Cleaning files in: {current_dir}")
    
    # Remove metadata files
    removed_count = remove_metadata_files(current_dir)
    
    # Show remaining files
    show_remaining_files(current_dir)
    
    print(f"\n=== Cleanup Summary ===")
    print(f"Metadata files removed: {removed_count}")
    print(f"Cleanup completed successfully!")
    print(f"Only data files (produkt_n_stunde) remain in the directory.")

if __name__ == "__main__":
    main()
