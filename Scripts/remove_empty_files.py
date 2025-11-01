#!/usr/bin/env python3
"""
Script to remove empty CSV files (files with only headers and no data).
Performs the following operations:
1. Read each CSV file
2. Check if file has only header row (no data)
3. Delete files that are empty
4. Count and report deleted files
"""

import os
import pandas as pd
import glob
import argparse
import logging
from pathlib import Path

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('remove_empty_files.log'),
            logging.StreamHandler()
        ]
    )

def remove_empty_files(folder_path):
    """
    Remove all CSV files that contain only headers (no data rows)
    """
    logging.info("Starting to remove empty CSV files...")
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if not csv_files:
        logging.info("No CSV files found to process.")
        return
    
    logging.info(f"Found {len(csv_files)} CSV files to check.")
    
    deleted_files = []
    kept_files = 0
    
    for csv_file in csv_files:
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file, encoding='utf-8')
            
            # Check if file has only header row (no data)
            if len(df) == 0:
                # File is completely empty (no rows at all)
                os.remove(csv_file)
                deleted_files.append(os.path.basename(csv_file))
                logging.info(f"Deleted empty file: {os.path.basename(csv_file)}")
            else:
                # File has data rows
                kept_files += 1
                logging.info(f"Kept file with data: {os.path.basename(csv_file)} ({len(df)} rows)")
                
        except Exception as e:
            logging.error(f"Error processing {csv_file}: {str(e)}")
    
    # Summary report
    logging.info("=" * 60)
    logging.info("EMPTY FILE REMOVAL SUMMARY:")
    logging.info(f"Total files checked: {len(csv_files)}")
    logging.info(f"Files deleted: {len(deleted_files)}")
    logging.info(f"Files kept: {kept_files}")
    
    if deleted_files:
        logging.info("Deleted files:")
        for deleted_file in deleted_files:
            logging.info(f"  - {deleted_file}")
    
    logging.info("=" * 60)
    logging.info("Empty file removal completed successfully!")

def main():
    """Main function to execute the removal"""
    parser = argparse.ArgumentParser(description='Remove empty CSV files')
    parser.add_argument('--folder', '-f', 
                       default='extreme_wind',
                       help='Path to the folder containing the CSV files (default: extreme_wind)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Get the absolute path
    folder_path = os.path.abspath(args.folder)
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        logging.error(f"Folder does not exist: {folder_path}")
        return
    
    logging.info(f"Starting removal of empty files in folder: {folder_path}")
    logging.info("=" * 60)
    
    try:
        # Execute the removal
        remove_empty_files(folder_path)
        
    except Exception as e:
        logging.error(f"An error occurred during removal: {str(e)}")
        raise

if __name__ == "__main__":
    main()
