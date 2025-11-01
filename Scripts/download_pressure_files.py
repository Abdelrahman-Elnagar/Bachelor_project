#!/usr/bin/env python3
"""
Script to extract href links from pressure HTML file and download them.
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
            logging.FileHandler('download_pressure_files.log'),
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
            # Filter for downloadable files (zip and txt files)
            if href.endswith('.zip') or href.endswith('.txt'):
                links.append(href)
        
        logging.info(f"Found {len(links)} downloadable files")
        return links
        
    except Exception as e:
        logging.error(f"Error reading HTML file: {str(e)}")
        return []

def download_file(url, download_dir):
    """
    Download a single file from URL
    """
    try:
        # Get filename from URL
        filename = os.path.basename(urlparse(url).path)
        filepath = os.path.join(download_dir, filename)
        
        # Skip if file already exists
        if os.path.exists(filepath):
            logging.info(f"File already exists, skipping: {filename}")
            return True
        
        # Download file
        logging.info(f"Downloading: {filename}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(filepath, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        
        logging.info(f"Successfully downloaded: {filename}")
        return True
        
    except Exception as e:
        logging.error(f"Error downloading {url}: {str(e)}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Download files from pressure HTML index')
    parser.add_argument('--html-file', default='pressure/Index of _climate_environment_CDC_observations_germany_climate_hourly_pressure_historical_.html',
                      help='Path to HTML file containing links')
    parser.add_argument('--download-dir', default='pressure',
                      help='Directory to download files to')
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Check if HTML file exists
    if not os.path.exists(args.html_file):
        logging.error(f"HTML file not found: {args.html_file}")
        return
    
    # Create download directory if it doesn't exist
    os.makedirs(args.download_dir, exist_ok=True)
    
    # Extract href links
    links = extract_href_links(args.html_file)
    
    if not links:
        logging.warning("No downloadable files found")
        return
    
    # Download files
    logging.info(f"Starting download of {len(links)} files...")
    successful_downloads = 0
    failed_downloads = 0
    
    for i, link in enumerate(links, 1):
        logging.info(f"Processing file {i}/{len(links)}")
        if download_file(link, args.download_dir):
            successful_downloads += 1
        else:
            failed_downloads += 1
    
    # Summary
    logging.info(f"Download completed!")
    logging.info(f"Successful downloads: {successful_downloads}")
    logging.info(f"Failed downloads: {failed_downloads}")
    logging.info(f"Total files processed: {len(links)}")

if __name__ == "__main__":
    main()
