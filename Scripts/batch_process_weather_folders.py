#!/usr/bin/env python3
"""
Batch processing script to run weather data processing on multiple folders sequentially.
This script will process the following folders:
- visibility
- weather_phenomena  
- wind
- wind_synop

Each folder will be processed using the process_weather_data.py script.
"""

import os
import sys
import subprocess
import logging
import argparse
from datetime import datetime
import time

def setup_logging():
    """Setup logging configuration for batch processing"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('batch_weather_processing.log'),
            logging.StreamHandler()
        ]
    )

def run_weather_processing(folder_path, script_path):
    """
    Run the weather data processing script on a specific folder
    
    Args:
        folder_path (str): Path to the folder to process
        script_path (str): Path to the process_weather_data.py script
    
    Returns:
        bool: True if successful, False otherwise
    """
    logging.info(f"Starting processing of folder: {folder_path}")
    logging.info("=" * 80)
    
    try:
        # Check if folder exists
        if not os.path.exists(folder_path):
            logging.error(f"Folder does not exist: {folder_path}")
            return False
        
        # Check if script exists
        if not os.path.exists(script_path):
            logging.error(f"Processing script does not exist: {script_path}")
            return False
        
        # Run the processing script
        cmd = [sys.executable, script_path, '--folder', folder_path]
        
        logging.info(f"Executing command: {' '.join(cmd)}")
        
        # Run the subprocess and capture output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout per folder
        )
        
        # Log the output
        if result.stdout:
            logging.info(f"STDOUT for {folder_path}:\n{result.stdout}")
        
        if result.stderr:
            logging.warning(f"STDERR for {folder_path}:\n{result.stderr}")
        
        # Check return code
        if result.returncode == 0:
            logging.info(f"Successfully completed processing of folder: {folder_path}")
            return True
        else:
            logging.error(f"Processing failed for folder {folder_path} with return code: {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        logging.error(f"Processing timed out for folder: {folder_path}")
        return False
    except Exception as e:
        logging.error(f"Error processing folder {folder_path}: {str(e)}")
        return False

def main():
    """Main function to execute batch processing"""
    parser = argparse.ArgumentParser(description='Batch process weather data folders')
    parser.add_argument('--base-dir', '-b', 
                       default='.',
                       help='Base directory containing the weather data folders (default: current directory)')
    parser.add_argument('--script-path', '-s',
                       default='Scripts/process_weather_data.py',
                       help='Path to the process_weather_data.py script (default: Scripts/process_weather_data.py)')
    parser.add_argument('--folders', '-f',
                       nargs='+',
                       default=['visibility', 'weather_phenomena', 'wind', 'wind_synop'],
                       help='List of folders to process (default: visibility weather_phenomena wind wind_synop)')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip folders that already have processed CSV files')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Get absolute paths
    base_dir = os.path.abspath(args.base_dir)
    script_path = os.path.abspath(args.script_path)
    
    # Check if base directory exists
    if not os.path.exists(base_dir):
        logging.error(f"Base directory does not exist: {base_dir}")
        return 1
    
    # Check if script exists
    if not os.path.exists(script_path):
        logging.error(f"Processing script does not exist: {script_path}")
        return 1
    
    logging.info(f"Starting batch processing at: {datetime.now()}")
    logging.info(f"Base directory: {base_dir}")
    logging.info(f"Script path: {script_path}")
    logging.info(f"Folders to process: {args.folders}")
    logging.info("=" * 80)
    
    # Track results
    successful_folders = []
    failed_folders = []
    skipped_folders = []
    
    # Process each folder
    for i, folder_name in enumerate(args.folders, 1):
        folder_path = os.path.join(base_dir, folder_name)
        
        logging.info(f"\nProcessing folder {i}/{len(args.folders)}: {folder_name}")
        logging.info("-" * 60)
        
        # Check if folder exists
        if not os.path.exists(folder_path):
            logging.warning(f"Folder does not exist, skipping: {folder_path}")
            skipped_folders.append(folder_name)
            continue
        
        # Check if we should skip existing processed files
        if args.skip_existing:
            csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
            if csv_files:
                logging.info(f"Found {len(csv_files)} existing CSV files, skipping folder: {folder_name}")
                skipped_folders.append(folder_name)
                continue
        
        # Run the processing
        start_time = time.time()
        success = run_weather_processing(folder_path, script_path)
        end_time = time.time()
        
        processing_time = end_time - start_time
        logging.info(f"Processing time for {folder_name}: {processing_time:.2f} seconds")
        
        if success:
            successful_folders.append(folder_name)
        else:
            failed_folders.append(folder_name)
        
        # Add a small delay between folders to be respectful
        if i < len(args.folders):
            logging.info("Waiting 5 seconds before processing next folder...")
            time.sleep(5)
    
    # Print summary
    logging.info("\n" + "=" * 80)
    logging.info("BATCH PROCESSING SUMMARY")
    logging.info("=" * 80)
    logging.info(f"Total folders processed: {len(args.folders)}")
    logging.info(f"Successful: {len(successful_folders)}")
    logging.info(f"Failed: {len(failed_folders)}")
    logging.info(f"Skipped: {len(skipped_folders)}")
    
    if successful_folders:
        logging.info(f"\nSuccessful folders: {', '.join(successful_folders)}")
    
    if failed_folders:
        logging.info(f"\nFailed folders: {', '.join(failed_folders)}")
    
    if skipped_folders:
        logging.info(f"\nSkipped folders: {', '.join(skipped_folders)}")
    
    logging.info(f"\nBatch processing completed at: {datetime.now()}")
    
    # Return appropriate exit code
    if failed_folders:
        return 1
    else:
        return 0

if __name__ == "__main__":
    sys.exit(main())
