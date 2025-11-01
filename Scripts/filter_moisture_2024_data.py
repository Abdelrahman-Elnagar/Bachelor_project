#!/usr/bin/env python3
"""
Script to filter moisture data to only include 2024 records and remove files with no 2024 data.
Performs the following operations:
1. Read each CSV file
2. Filter data to only include 2024 records
3. If file has no 2024 data, delete the entire file
4. If file has 2024 data, save the filtered data
5. Count and report results
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
            logging.FileHandler('filter_moisture_2024_data.log'),
            logging.StreamHandler()
        ]
    )

def filter_2024_data_and_remove_empty_files(folder_path):
    """
    Filter all CSV files to only include 2024 records and remove files with no 2024 data
    """
    logging.info("Starting to filter data to only include 2024 records...")
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if not csv_files:
        logging.info("No CSV files found to process.")
        return
    
    logging.info(f"Found {len(csv_files)} CSV files to process.")
    
    files_with_2024_data = 0
    files_removed = 0
    total_records_before = 0
    total_records_after = 0
    removed_files = []
    
    for csv_file in csv_files:
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file, encoding='utf-8')
            
            if len(df) == 0:
                # Empty file, remove it
                os.remove(csv_file)
                files_removed += 1
                removed_files.append(os.path.basename(csv_file))
                logging.info(f"Removed empty file: {os.path.basename(csv_file)}")
                continue
            
            # Check if MESS_DATUM column exists
            if 'MESS_DATUM' not in df.columns:
                logging.warning(f"MESS_DATUM column not found in {os.path.basename(csv_file)}")
                continue
            
            # Convert MESS_DATUM to datetime
            df['MESS_DATUM'] = pd.to_datetime(df['MESS_DATUM'])
            
            # Filter for 2024 data
            df_2024 = df[df['MESS_DATUM'].dt.year == 2024]
            
            total_records_before += len(df)
            
            if len(df_2024) == 0:
                # No 2024 data, remove the file
                os.remove(csv_file)
                files_removed += 1
                removed_files.append(os.path.basename(csv_file))
                logging.info(f"Removed file with no 2024 data: {os.path.basename(csv_file)}")
            else:
                # Has 2024 data, save the filtered data
                df_2024.to_csv(csv_file, index=False, encoding='utf-8')
                total_records_after += len(df_2024)
                files_with_2024_data += 1
                logging.info(f"File {os.path.basename(csv_file)}: {len(df)} -> {len(df_2024)} records")
                
        except Exception as e:
            logging.error(f"Error processing {csv_file}: {str(e)}")
    
    # Summary report
    logging.info("=" * 60)
    logging.info("2024 DATA FILTERING SUMMARY:")
    logging.info(f"Total files processed: {len(csv_files)}")
    logging.info(f"Files with 2024 data (kept): {files_with_2024_data}")
    logging.info(f"Files removed (no 2024 data): {files_removed}")
    logging.info(f"Total records before filtering: {total_records_before:,}")
    logging.info(f"Total records after filtering: {total_records_after:,}")
    logging.info(f"Records removed: {total_records_before - total_records_after:,}")
    
    if removed_files:
        logging.info("Removed files:")
        for removed_file in removed_files:
            logging.info(f"  - {removed_file}")
    
    logging.info("=" * 60)
    logging.info("2024 data filtering completed successfully!")

def main():
    """Main function to execute the filtering"""
    parser = argparse.ArgumentParser(description='Filter moisture data to only include 2024 records and remove files with no 2024 data')
    parser.add_argument('--folder', '-f', 
                       default='moisture',
                       help='Path to the folder containing the CSV files (default: moisture)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Get the absolute path
    folder_path = os.path.abspath(args.folder)
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        logging.error(f"Folder does not exist: {folder_path}")
        return
    
    logging.info(f"Starting filtering of folder: {folder_path}")
    logging.info("=" * 60)
    
    try:
        # Execute the filtering
        filter_2024_data_and_remove_empty_files(folder_path)
        
    except Exception as e:
        logging.error(f"An error occurred during filtering: {str(e)}")
        raise

if __name__ == "__main__":
    main()
