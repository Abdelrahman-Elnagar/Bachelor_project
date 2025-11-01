#!/usr/bin/env python3
"""
Script to download all dew point data files from the DWD (German Weather Service) website.
This script extracts URLs from the HTML index file and downloads all the zip files.
"""

import re
import os
import requests
from urllib.parse import urlparse
from pathlib import Path
import time

def extract_download_urls(html_file_path):
    """Extract all download URLs from the HTML file."""
    urls = []
    
    with open(html_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Find all href links that point to zip files
    pattern = r'href="(https://opendata\.dwd\.de/climate_environment/CDC/observations_germany/climate/hourly/dew_point/historical/[^"]*\.zip)"'
    matches = re.findall(pattern, content)
    
    for match in matches:
        urls.append(match)
    
    return urls

def download_file(url, download_dir):
    """Download a single file from URL to the specified directory."""
    try:
        # Parse the URL to get the filename
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        filepath = os.path.join(download_dir, filename)
        
        # Skip if file already exists
        if os.path.exists(filepath):
            print(f"File {filename} already exists, skipping...")
            return True
        
        print(f"Downloading {filename}...")
        
        # Download the file
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Write the file
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Successfully downloaded {filename}")
        return True
        
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return False

def main():
    # Set up paths
    html_file = "dew_point/Index of _climate_environment_CDC_observations_germany_climate_hourly_dew_point_historical_.html"
    download_dir = "dew_point"
    
    # Create download directory if it doesn't exist
    os.makedirs(download_dir, exist_ok=True)
    
    print("Extracting download URLs from HTML file...")
    urls = extract_download_urls(html_file)
    
    print(f"Found {len(urls)} files to download")
    
    # Download all files
    successful_downloads = 0
    failed_downloads = 0
    
    for i, url in enumerate(urls, 1):
        print(f"\nProgress: {i}/{len(urls)}")
        
        if download_file(url, download_dir):
            successful_downloads += 1
        else:
            failed_downloads += 1
        
        # Add a small delay to be respectful to the server
        time.sleep(0.5)
    
    print(f"\nDownload completed!")
    print(f"Successful downloads: {successful_downloads}")
    print(f"Failed downloads: {failed_downloads}")

if __name__ == "__main__":
    main()

