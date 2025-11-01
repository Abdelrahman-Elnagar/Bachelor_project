#!/usr/bin/env python3
"""
Script to extract href links from precipitation HTML file and download them.
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
            logging.FileHandler('download_precipitation_files.log'),
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
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            # Filter for zip and txt files, exclude parent directory links
            if href.endswith('.zip') or href.endswith('.txt'):
                if not href.startswith('../'):
                    links.append(href)
        
        logging.info(f"Found {len(links)} downloadable files")
        return links
        
    except Exception as e:
        logging.error(f"Error reading HTML file: {str(e)}")
        return []

def download_file(url, local_path):
    """
    Download a file from URL to local path
    """
    try:
        logging.info(f"Downloading: {os.path.basename(local_path)}")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(local_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        
        logging.info(f"Successfully downloaded: {os.path.basename(local_path)}")
        return True
        
    except Exception as e:
        logging.error(f"Error downloading {url}: {str(e)}")
        return False

def download_precipitation_files(html_file_path):
    """
    Download all precipitation files from HTML index
    """
    logging.info("Starting precipitation files download...")
    
    # Extract links from HTML file
    links = extract_href_links(html_file_path)
    
    if not links:
        logging.warning("No downloadable files found in HTML")
        return
    
    # Get directory path
    directory = os.path.dirname(html_file_path)
    
    # Download files
    downloaded_count = 0
    failed_count = 0
    
    for link in links:
        filename = os.path.basename(link)
        local_path = os.path.join(directory, filename)
        
        # Skip if file already exists
        if os.path.exists(local_path):
            logging.info(f"File already exists, skipping: {filename}")
            continue
        
        if download_file(link, local_path):
            downloaded_count += 1
        else:
            failed_count += 1
    
    logging.info(f"Download completed!")
    logging.info(f"Files downloaded: {downloaded_count}")
    logging.info(f"Files failed: {failed_count}")
    logging.info(f"Total files processed: {len(links)}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Download precipitation files from HTML index')
    parser.add_argument('--html-file', 
                       default='precipitation/Index of _climate_environment_CDC_observations_germany_climate_hourly_precipitation_historical_.html',
                       help='Path to HTML file containing download links')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Check if HTML file exists
    if not os.path.exists(args.html_file):
        logging.error(f"HTML file not found: {args.html_file}")
        return
    
    # Download files
    download_precipitation_files(args.html_file)

if __name__ == "__main__":
    main()
