#!/usr/bin/env python3
"""
Script to download all precipitation data files from the HTML index.
Downloads all ZIP files referenced in the precipitation HTML index file.
"""

import os
import re
import requests
from urllib.parse import urlparse
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_download_urls(html_file_path):
    """Extract all download URLs from the HTML file."""
    urls = []
    
    with open(html_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Find all href links that point to ZIP files
    pattern = r'href="(https://opendata\.dwd\.de/climate_environment/CDC/observations_germany/climate/hourly/precipitation/historical/[^"]*\.zip)"'
    matches = re.findall(pattern, content)
    
    for match in matches:
        urls.append(match)
    
    logger.info(f"Found {len(urls)} download URLs")
    return urls

def download_file(url, download_dir):
    """Download a single file from URL to download directory."""
    try:
        # Parse filename from URL
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        file_path = os.path.join(download_dir, filename)
        
        # Skip if file already exists
        if os.path.exists(file_path):
            logger.info(f"File {filename} already exists, skipping")
            return True, filename, "Already exists"
        
        # Download the file
        logger.info(f"Downloading {filename}...")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Write file in chunks
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        file_size = os.path.getsize(file_path)
        logger.info(f"Successfully downloaded {filename} ({file_size:,} bytes)")
        return True, filename, f"Downloaded ({file_size:,} bytes)"
        
    except Exception as e:
        logger.error(f"Failed to download {url}: {str(e)}")
        return False, os.path.basename(urlparse(url).path), str(e)

def download_all_files(html_file_path, download_dir="precipitation_downloads", max_workers=5):
    """Download all files from the HTML index."""
    
    # Create download directory
    os.makedirs(download_dir, exist_ok=True)
    logger.info(f"Created download directory: {download_dir}")
    
    # Extract URLs
    urls = extract_download_urls(html_file_path)
    
    if not urls:
        logger.error("No URLs found to download")
        return
    
    # Download files with threading
    successful_downloads = 0
    failed_downloads = 0
    
    logger.info(f"Starting download of {len(urls)} files with {max_workers} workers...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_url = {
            executor.submit(download_file, url, download_dir): url 
            for url in urls
        }
        
        # Process completed downloads
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                success, filename, message = future.result()
                if success:
                    successful_downloads += 1
                else:
                    failed_downloads += 1
                    logger.warning(f"Failed: {filename} - {message}")
            except Exception as e:
                failed_downloads += 1
                logger.error(f"Exception for {url}: {str(e)}")
    
    # Summary
    logger.info(f"Download completed!")
    logger.info(f"Successful downloads: {successful_downloads}")
    logger.info(f"Failed downloads: {failed_downloads}")
    logger.info(f"Total files processed: {len(urls)}")

def main():
    """Main function."""
    html_file = "precipitation/Index of _climate_environment_CDC_observations_germany_climate_hourly_precipitation_historical_.html"
    
    if not os.path.exists(html_file):
        logger.error(f"HTML file not found: {html_file}")
        return
    
    logger.info("Starting precipitation data download...")
    download_all_files(html_file)
    logger.info("Download process completed!")

if __name__ == "__main__":
    main()


