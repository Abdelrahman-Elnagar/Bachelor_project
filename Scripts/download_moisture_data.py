#!/usr/bin/env python3
"""
Script to download all moisture data files from the HTML index page.
"""

import re
import requests
import os
from urllib.parse import urlparse
from pathlib import Path
import time

def extract_download_urls(html_file_path):
    """Extract all download URLs from the HTML file."""
    urls = []
    
    with open(html_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all href links that point to zip files
    pattern = r'href="(https://opendata\.dwd\.de/climate_environment/CDC/observations_germany/climate/hourly/moisture/historical/[^"]*\.zip)"'
    matches = re.findall(pattern, content)
    
    for url in matches:
        # Extract filename from URL
        filename = url.split('/')[-1]
        urls.append((url, filename))
    
    return urls

def download_file(url, filename, download_dir):
    """Download a single file."""
    filepath = os.path.join(download_dir, filename)
    
    # Skip if file already exists
    if os.path.exists(filepath):
        print(f"File {filename} already exists, skipping...")
        return True
    
    try:
        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Successfully downloaded {filename}")
        return True
        
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        return False

def main():
    # Set up paths
    html_file = "moisture/Index of _climate_environment_CDC_observations_germany_climate_hourly_moisture_historical_.html"
    download_dir = "moisture_downloads"
    
    # Create download directory if it doesn't exist
    os.makedirs(download_dir, exist_ok=True)
    
    print("Extracting download URLs from HTML file...")
    urls = extract_download_urls(html_file)
    
    print(f"Found {len(urls)} files to download")
    
    # Download files
    successful_downloads = 0
    failed_downloads = 0
    
    for i, (url, filename) in enumerate(urls, 1):
        print(f"\nProgress: {i}/{len(urls)}")
        
        if download_file(url, filename, download_dir):
            successful_downloads += 1
        else:
            failed_downloads += 1
        
        # Add a small delay to be respectful to the server
        time.sleep(0.5)
    
    print(f"\nDownload complete!")
    print(f"Successfully downloaded: {successful_downloads}")
    print(f"Failed downloads: {failed_downloads}")
    print(f"Files saved to: {download_dir}")

if __name__ == "__main__":
    main()


