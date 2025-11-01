#!/usr/bin/env python3
"""
Script to extract href links from HTML file and download them.
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
            logging.FileHandler('download_moisture_files.log'),
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
    except UnicodeDecodeError:
        # Try with different encoding if UTF-8 fails
        with open(html_file_path, 'r', encoding='windows-1252') as file:
            content = file.read()
    
    # Parse HTML content
    soup = BeautifulSoup(content, 'html.parser')
    
    # Find all href links
    links = []
    for link in soup.find_all('a', href=True):
        href = link['href']
        # Skip parent directory links
        if href != '../' and not href.endswith('/'):
            links.append(href)
    
    logging.info(f"Found {len(links)} href links")
    return links

def download_file(url, download_dir, filename=None):
    """
    Download a file from URL to the specified directory
    """
    try:
        # Create filename from URL if not provided
        if not filename:
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path)
        
        filepath = os.path.join(download_dir, filename)
        
        # Skip if file already exists
        if os.path.exists(filepath):
            logging.info(f"File already exists, skipping: {filename}")
            return True
        
        logging.info(f"Downloading: {filename}")
        
        # Download the file
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Save the file
        with open(filepath, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        
        file_size = os.path.getsize(filepath)
        logging.info(f"Downloaded: {filename} ({file_size:,} bytes)")
        return True
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to download {url}: {str(e)}")
        return False
    except Exception as e:
        logging.error(f"Error downloading {url}: {str(e)}")
        return False

def download_all_files(html_file_path):
    """
    Extract href links and download all files
    """
    # Get the directory containing the HTML file
    html_dir = os.path.dirname(os.path.abspath(html_file_path))
    
    logging.info(f"Download directory: {html_dir}")
    
    # Extract href links
    links = extract_href_links(html_file_path)
    
    if not links:
        logging.warning("No links found in HTML file")
        return
    
    # Filter for downloadable files (zip and txt files)
    downloadable_links = []
    for link in links:
        if link.endswith('.zip') or link.endswith('.txt'):
            downloadable_links.append(link)
    
    logging.info(f"Found {len(downloadable_links)} downloadable files")
    
    if not downloadable_links:
        logging.warning("No downloadable files found")
        return
    
    # Download all files
    successful_downloads = 0
    failed_downloads = 0
    
    for i, link in enumerate(downloadable_links, 1):
        logging.info(f"Processing file {i}/{len(downloadable_links)}: {os.path.basename(link)}")
        
        if download_file(link, html_dir):
            successful_downloads += 1
        else:
            failed_downloads += 1
    
    # Summary report
    logging.info("=" * 60)
    logging.info("DOWNLOAD SUMMARY:")
    logging.info(f"Total files found: {len(downloadable_links)}")
    logging.info(f"Successful downloads: {successful_downloads}")
    logging.info(f"Failed downloads: {failed_downloads}")
    logging.info(f"Download directory: {html_dir}")
    logging.info("=" * 60)
    logging.info("Download process completed!")

def main():
    """Main function to execute the download"""
    parser = argparse.ArgumentParser(description='Download all href links from HTML file')
    parser.add_argument('--html-file', '-f', 
                       default='moisture/Index of _climate_environment_CDC_observations_germany_climate_hourly_moisture_historical_.html',
                       help='Path to the HTML file containing href links')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Check if HTML file exists
    if not os.path.exists(args.html_file):
        logging.error(f"HTML file does not exist: {args.html_file}")
        return
    
    logging.info(f"Starting download process for: {args.html_file}")
    logging.info("=" * 60)
    
    try:
        # Execute the download
        download_all_files(args.html_file)
        
    except Exception as e:
        logging.error(f"An error occurred during download: {str(e)}")
        raise

if __name__ == "__main__":
    main()
