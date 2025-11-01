#!/usr/bin/env python3
"""
Script to process extreme wind data files sequentially.
Performs the following operations in order:
1. Unzip all files
2. Delete all zipped files
3. Delete all HTML files
4. Delete all txt files that start with "meta" in the filename
5. Convert all remaining txt files to CSV files
6. Update MESS_DATUM column to proper date format
7. Rename all files with their STATIONS_ID value
8. Remove 'eor' column if it exists
"""

import os
import zipfile
import pandas as pd
import glob
from pathlib import Path
import argparse
import logging
from datetime import datetime

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('extreme_wind_processing.log'),
            logging.StreamHandler()
        ]
    )

def unzip_files(folder_path):
    """
    Step 1: Unzip all zip files in the specified folder
    """
    logging.info("Step 1: Starting to unzip all files...")
    zip_files = glob.glob(os.path.join(folder_path, "*.zip"))
    
    if not zip_files:
        logging.info("No zip files found in the folder.")
        return
    
    logging.info(f"Found {len(zip_files)} zip files to extract.")
    
    for zip_file in zip_files:
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(folder_path)
            logging.info(f"Successfully extracted: {os.path.basename(zip_file)}")
        except Exception as e:
            logging.error(f"Error extracting {zip_file}: {str(e)}")
    
    logging.info("Step 1 completed: All files unzipped.")

def delete_zip_files(folder_path):
    """
    Step 2: Delete all zip files
    """
    logging.info("Step 2: Starting to delete zip files...")
    zip_files = glob.glob(os.path.join(folder_path, "*.zip"))
    
    if not zip_files:
        logging.info("No zip files found to delete.")
        return
    
    logging.info(f"Found {len(zip_files)} zip files to delete.")
    
    for zip_file in zip_files:
        try:
            os.remove(zip_file)
            logging.info(f"Successfully deleted: {os.path.basename(zip_file)}")
        except Exception as e:
            logging.error(f"Error deleting {zip_file}: {str(e)}")
    
    logging.info("Step 2 completed: All zip files deleted.")

def delete_html_files(folder_path):
    """
    Step 3: Delete all HTML files
    """
    logging.info("Step 3: Starting to delete HTML files...")
    html_files = glob.glob(os.path.join(folder_path, "*.html"))
    
    if not html_files:
        logging.info("No HTML files found to delete.")
        return
    
    logging.info(f"Found {len(html_files)} HTML files to delete.")
    
    for html_file in html_files:
        try:
            os.remove(html_file)
            logging.info(f"Successfully deleted: {os.path.basename(html_file)}")
        except Exception as e:
            logging.error(f"Error deleting {html_file}: {str(e)}")
    
    logging.info("Step 3 completed: All HTML files deleted.")

def delete_meta_txt_files(folder_path):
    """
    Step 4: Delete all txt files that start with "meta" in the filename
    """
    logging.info("Step 4: Starting to delete meta*.txt files...")
    meta_files = glob.glob(os.path.join(folder_path, "meta*.txt"))
    
    if not meta_files:
        logging.info("No meta*.txt files found to delete.")
        return
    
    logging.info(f"Found {len(meta_files)} meta*.txt files to delete.")
    
    for meta_file in meta_files:
        try:
            os.remove(meta_file)
            logging.info(f"Successfully deleted: {os.path.basename(meta_file)}")
        except Exception as e:
            logging.error(f"Error deleting {meta_file}: {str(e)}")
    
    logging.info("Step 4 completed: All meta*.txt files deleted.")

def convert_txt_to_csv(folder_path):
    """
    Step 5: Convert all remaining txt files to CSV files
    """
    logging.info("Step 5: Starting to convert txt files to CSV...")
    txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
    
    if not txt_files:
        logging.info("No txt files found to convert.")
        return
    
    logging.info(f"Found {len(txt_files)} txt files to convert.")
    
    for txt_file in txt_files:
        try:
            # Read the txt file with semicolon separator
            df = pd.read_csv(txt_file, sep=';', encoding='utf-8')
            
            # Create CSV filename
            csv_file = txt_file.replace('.txt', '.csv')
            
            # Save as CSV
            df.to_csv(csv_file, index=False, encoding='utf-8')
            
            # Delete the original txt file
            os.remove(txt_file)
            
            logging.info(f"Successfully converted: {os.path.basename(txt_file)} -> {os.path.basename(csv_file)}")
        except Exception as e:
            logging.error(f"Error converting {txt_file}: {str(e)}")
    
    logging.info("Step 5 completed: All txt files converted to CSV.")

def format_mess_datum_column(folder_path):
    """
    Step 6: Update MESS_DATUM column to proper date format
    """
    logging.info("Step 6: Starting to format MESS_DATUM column...")
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if not csv_files:
        logging.info("No CSV files found to process.")
        return
    
    logging.info(f"Found {len(csv_files)} CSV files to process.")
    
    for csv_file in csv_files:
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file, encoding='utf-8')
            
            # Check if MESS_DATUM column exists
            if 'MESS_DATUM' not in df.columns:
                logging.warning(f"MESS_DATUM column not found in {os.path.basename(csv_file)}")
                continue
            
            # Convert MESS_DATUM from YYYYMMDDHH format to proper datetime
            # The format appears to be YYYYMMDDHH (e.g., 2023101119)
            df['MESS_DATUM'] = pd.to_datetime(df['MESS_DATUM'], format='%Y%m%d%H', errors='coerce')
            
            # Save the updated CSV
            df.to_csv(csv_file, index=False, encoding='utf-8')
            
            logging.info(f"Successfully formatted MESS_DATUM in: {os.path.basename(csv_file)}")
        except Exception as e:
            logging.error(f"Error formatting MESS_DATUM in {csv_file}: {str(e)}")
    
    logging.info("Step 6 completed: All MESS_DATUM columns formatted.")

def rename_files_with_stations_id(folder_path):
    """
    Step 7: Rename all CSV files using STATIONS_ID values
    """
    logging.info("Step 7: Starting to rename files with STATIONS_ID...")
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if not csv_files:
        logging.info("No CSV files found to rename.")
        return
    
    logging.info(f"Found {len(csv_files)} CSV files to rename.")
    
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
            
            # Create new filename
            new_filename = f"{stations_id}.csv"
            new_filepath = os.path.join(folder_path, new_filename)
            
            # Check if target file already exists and is different from current file
            if os.path.exists(new_filepath) and new_filepath != csv_file:
                logging.warning(f"File {new_filename} already exists, skipping {os.path.basename(csv_file)}")
                continue
            
            # Rename the file if it's different from the new name
            if new_filepath != csv_file:
                os.rename(csv_file, new_filepath)
                logging.info(f"Renamed: {os.path.basename(csv_file)} -> {new_filename}")
            else:
                logging.info(f"File already correctly named: {os.path.basename(csv_file)}")
                
        except Exception as e:
            logging.error(f"Error renaming {csv_file}: {str(e)}")
    
    logging.info("Step 7 completed: All files renamed with STATIONS_ID.")

def remove_eor_column(folder_path):
    """
    Step 8: Remove 'eor' column if it exists
    """
    logging.info("Step 8: Starting to remove 'eor' column...")
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if not csv_files:
        logging.info("No CSV files found to process.")
        return
    
    logging.info(f"Found {len(csv_files)} CSV files to process.")
    
    for csv_file in csv_files:
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file, encoding='utf-8')
            
            # Remove 'eor' column if it exists
            if 'eor' in df.columns:
                df = df.drop('eor', axis=1)
                
                # Save the updated CSV
                df.to_csv(csv_file, index=False, encoding='utf-8')
                logging.info(f"Removed 'eor' column from: {os.path.basename(csv_file)}")
            else:
                logging.info(f"No 'eor' column found in: {os.path.basename(csv_file)}")
                
        except Exception as e:
            logging.error(f"Error removing 'eor' column from {csv_file}: {str(e)}")
    
    logging.info("Step 8 completed: All 'eor' columns removed.")

def main():
    """Main function to execute all processing steps sequentially"""
    parser = argparse.ArgumentParser(description='Process extreme wind data files')
    parser.add_argument('--folder', '-f', 
                       default='extreme_wind',
                       help='Path to the folder containing the data files (default: extreme_wind)')
    
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
        # Execute all steps sequentially
        unzip_files(folder_path)
        logging.info("-" * 40)
        
        delete_zip_files(folder_path)
        logging.info("-" * 40)
        
        delete_html_files(folder_path)
        logging.info("-" * 40)
        
        delete_meta_txt_files(folder_path)
        logging.info("-" * 40)
        
        convert_txt_to_csv(folder_path)
        logging.info("-" * 40)
        
        format_mess_datum_column(folder_path)
        logging.info("-" * 40)
        
        rename_files_with_stations_id(folder_path)
        logging.info("-" * 40)
        
        remove_eor_column(folder_path)
        logging.info("-" * 40)
        
        logging.info("=" * 60)
        logging.info("All processing steps completed successfully!")
        
    except Exception as e:
        logging.error(f"An error occurred during processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()
