#!/usr/bin/env python3
"""
Script to filter extreme wind data to only include 2024 records.
Performs the following operations:
1. Read each CSV file
2. Filter data to only include records from 2024
3. Save the filtered data back to the same file
4. Keep files even if they become empty
5. Count and report empty files
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
            logging.FileHandler('filter_2024_data.log'),
            logging.StreamHandler()
        ]
    )

def filter_2024_data(folder_path):
    """
    Filter all CSV files to only include 2024 data
    """
    logging.info("Starting to filter data to only include 2024 records...")
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if not csv_files:
        logging.info("No CSV files found to process.")
        return
    
    logging.info(f"Found {len(csv_files)} CSV files to process.")
    
    empty_files = []
    processed_files = 0
    total_records_before = 0
    total_records_after = 0
    
    for csv_file in csv_files:
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file, encoding='utf-8')
            
            # Check if MESS_DATUM column exists
            if 'MESS_DATUM' not in df.columns:
                logging.warning(f"MESS_DATUM column not found in {os.path.basename(csv_file)}")
                continue
            
            # Convert MESS_DATUM to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(df['MESS_DATUM']):
                df['MESS_DATUM'] = pd.to_datetime(df['MESS_DATUM'], errors='coerce')
            
            # Count records before filtering
            records_before = len(df)
            total_records_before += records_before
            
            # Filter to only include 2024 data
            df_2024 = df[df['MESS_DATUM'].dt.year == 2024]
            
            # Count records after filtering
            records_after = len(df_2024)
            total_records_after += records_after
            
            # Save the filtered data back to the same file
            df_2024.to_csv(csv_file, index=False, encoding='utf-8')
            
            # Check if file is now empty
            if records_after == 0:
                empty_files.append(os.path.basename(csv_file))
                logging.info(f"File {os.path.basename(csv_file)}: {records_before} -> {records_after} records (EMPTY)")
            else:
                logging.info(f"File {os.path.basename(csv_file)}: {records_before} -> {records_after} records")
            
            processed_files += 1
            
        except Exception as e:
            logging.error(f"Error processing {csv_file}: {str(e)}")
    
    # Summary report
    logging.info("=" * 60)
    logging.info("FILTERING SUMMARY:")
    logging.info(f"Total files processed: {processed_files}")
    logging.info(f"Total records before filtering: {total_records_before:,}")
    logging.info(f"Total records after filtering: {total_records_after:,}")
    logging.info(f"Records removed: {total_records_before - total_records_after:,}")
    logging.info(f"Empty files (kept): {len(empty_files)}")
    
    if empty_files:
        logging.info("Empty files:")
        for empty_file in empty_files:
            logging.info(f"  - {empty_file}")
    
    logging.info("=" * 60)
    logging.info("2024 data filtering completed successfully!")

def main():
    """Main function to execute the filtering"""
    parser = argparse.ArgumentParser(description='Filter extreme wind data to only include 2024 records')
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
    
    logging.info(f"Starting filtering of folder: {folder_path}")
    logging.info("=" * 60)
    
    try:
        # Execute the filtering
        filter_2024_data(folder_path)
        
    except Exception as e:
        logging.error(f"An error occurred during filtering: {str(e)}")
        raise

if __name__ == "__main__":
    main()
