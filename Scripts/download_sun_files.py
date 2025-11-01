#!/usr/bin/env python3
"""
Script to extract href links from sun HTML file and download them.
Performs the following operations:
1. Parse HTML file to extract all href links
2. Filter for downloadable files (zip files and txt files)
3. Download all files to the same directory as the HTML file
4. Report download progress and results
"""

import os
import re
import requests
import argparse
import logging
from pathlib import Path
from urllib.parse import urlparse
from bs4 import BeautifulSoup

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('download_sun_files.log'),
            logging.StreamHandler()
        ]
    )

def extract_href_links(html_file_path):
    """
    Extract all href links from HTML file
    """
    logging.info(f"Reading HTML file: {html_file_path}")
    
    try:
        with open(html_file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Parse HTML content
        soup = BeautifulSoup(content, 'html.parser')
        
        # Find all href links
        links = soup.find_all('a', href=True)
        
        # Extract href values
        href_links = [link['href'] for link in links]
        
        # Filter for downloadable files (zip and txt files)
        downloadable_links = []
        for link in href_links:
            if link.endswith('.zip') or link.endswith('.txt'):
                downloadable_links.append(link)
        
        logging.info(f"Found {len(href_links)} total links")
        logging.info(f"Found {len(downloadable_links)} downloadable files")
        
        return downloadable_links
        
    except Exception as e:
        logging.error(f"Error reading HTML file: {str(e)}")
        return []

def download_file(url, download_path):
    """
    Download a single file from URL
    """
    try:
        logging.info(f"Downloading: {url}")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(download_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        
        logging.info(f"Downloaded successfully: {os.path.basename(download_path)}")
        return True
        
    except Exception as e:
        logging.error(f"Error downloading {url}: {str(e)}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Download sun files from HTML index')
    parser.add_argument('--html-file', default='sun/Index of _climate_environment_CDC_observations_germany_climate_hourly_sun_historical_.html',
                       help='Path to HTML file containing links')
    parser.add_argument('--download-dir', default='sun',
                       help='Directory to download files to')
    
    args = parser.parse_args()
    
    setup_logging()
    
    logging.info("Starting sun file download process...")
    
    # Check if HTML file exists
    if not os.path.exists(args.html_file):
        logging.error(f"HTML file not found: {args.html_file}")
        return
    
    # Create download directory if it doesn't exist
    os.makedirs(args.download_dir, exist_ok=True)
    
    # Extract href links
    downloadable_links = extract_href_links(args.html_file)
    
    if not downloadable_links:
        logging.error("No downloadable links found")
        return
    
    # Download files
    successful_downloads = 0
    failed_downloads = 0
    
    for link in downloadable_links:
        filename = os.path.basename(link)
        download_path = os.path.join(args.download_dir, filename)
        
        # Skip if file already exists
        if os.path.exists(download_path):
            logging.info(f"File already exists, skipping: {filename}")
            continue
        
        if download_file(link, download_path):
            successful_downloads += 1
        else:
            failed_downloads += 1
    
    # Summary
    logging.info("=" * 50)
    logging.info("DOWNLOAD SUMMARY")
    logging.info("=" * 50)
    logging.info(f"Total files to download: {len(downloadable_links)}")
    logging.info(f"Successful downloads: {successful_downloads}")
    logging.info(f"Failed downloads: {failed_downloads}")
    logging.info(f"Download directory: {args.download_dir}")
    
    if successful_downloads > 0:
        logging.info("Download process completed successfully!")
    else:
        logging.error("No files were downloaded successfully")

if __name__ == "__main__":
    main()
