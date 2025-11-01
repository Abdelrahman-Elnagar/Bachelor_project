#!/usr/bin/env python3
"""
Script to rename CSV files using STATIONS_ID and remove 'eor' column.
Performs the following operations:
1. Read each CSV file to get STATIONS_ID value
2. Rename file to STATIONS_ID.csv
3. Remove 'eor' column if it exists
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
            logging.FileHandler('rename_extreme_wind_files.log'),
            logging.StreamHandler()
        ]
    )

def rename_files_and_remove_eor(folder_path):
    """
    Rename CSV files using STATIONS_ID and remove 'eor' column
    """
    logging.info("Starting to rename files and remove 'eor' column...")
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if not csv_files:
        logging.info("No CSV files found to process.")
        return
    
    logging.info(f"Found {len(csv_files)} CSV files to process.")
    
    for csv_file in csv_files:
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file, encoding='utf-8')
            
            # Check if STATIONS_ID column exists
            if 'STATIONS_ID' not in df.columns:
                logging.warning(f"STATIONS_ID column not found in {os.path.basename(csv_file)}")
                continue
            
            # Get the STATIONS_ID value (should be the same for all rows)
            stations_id = df['STATIONS_ID'].iloc[0]
            
            # Remove 'eor' column if it exists
            if 'eor' in df.columns:
                df = df.drop('eor', axis=1)
                logging.info(f"Removed 'eor' column from {os.path.basename(csv_file)}")
            
            # Create new filename
            new_filename = f"{stations_id}.csv"
            new_filepath = os.path.join(folder_path, new_filename)
            
            # Check if target file already exists
            if os.path.exists(new_filepath) and new_filepath != csv_file:
                logging.warning(f"File {new_filename} already exists, skipping {os.path.basename(csv_file)}")
                continue
            
            # Save the updated CSV
            df.to_csv(new_filepath, index=False, encoding='utf-8')
            
            # Remove the original file if it's different from the new one
            if new_filepath != csv_file:
                os.remove(csv_file)
                logging.info(f"Renamed: {os.path.basename(csv_file)} -> {new_filename}")
            else:
                logging.info(f"Updated: {os.path.basename(csv_file)} (removed 'eor' column)")
                
        except Exception as e:
            logging.error(f"Error processing {csv_file}: {str(e)}")
    
    logging.info("File renaming and 'eor' column removal completed.")

def main():
    """Main function to execute the processing"""
    parser = argparse.ArgumentParser(description='Rename CSV files using STATIONS_ID and remove eor column')
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
    
    logging.info(f"Starting processing of folder: {folder_path}")
    logging.info("=" * 60)
    
    try:
        # Execute the processing
        rename_files_and_remove_eor(folder_path)
        
        logging.info("=" * 60)
        logging.info("Processing completed successfully!")
        
    except Exception as e:
        logging.error(f"An error occurred during processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()
