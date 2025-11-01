#!/usr/bin/env python3
"""
Script to download all pressure data files from the DWD (German Weather Service) website.
This script parses the HTML index file and downloads all pressure data files.
"""

import os
import re
import requests
import time
from urllib.parse import urlparse
from pathlib import Path

def extract_download_urls(html_file_path):
    """Extract all download URLs from the HTML file."""
    urls = []
    
    with open(html_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Find all href attributes that point to zip files
    pattern = r'href="(https://opendata\.dwd\.de[^"]*\.zip)"'
    matches = re.findall(pattern, content)
    
    for url in matches:
        urls.append(url)
    
    return urls

def download_file(url, download_dir):
    """Download a single file from URL to the specified directory."""
    try:
        # Extract filename from URL
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        filepath = os.path.join(download_dir, filename)
        
        # Skip if file already exists
        if os.path.exists(filepath):
            print(f"File already exists, skipping: {filename}")
            return True
        
        print(f"Downloading: {filename}")
        
        # Download the file
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Write file in chunks to handle large files
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        print(f"Successfully downloaded: {filename}")
        return True
        
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return False

def main():
    """Main function to download all pressure data files."""
    # Set up paths
    script_dir = Path(__file__).parent
    html_file = script_dir / "pressure" / "Index of _climate_environment_CDC_observations_germany_climate_hourly_pressure_historical_.html"
    download_dir = script_dir / "pressure_downloads"
    
    # Create download directory if it doesn't exist
    download_dir.mkdir(exist_ok=True)
    
    print(f"Reading HTML file: {html_file}")
    
    # Extract URLs from HTML file
    urls = extract_download_urls(html_file)
    print(f"Found {len(urls)} files to download")
    
    # Download files
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
    
    print(f"\nDownload Summary:")
    print(f"Successful downloads: {successful_downloads}")
    print(f"Failed downloads: {failed_downloads}")
    print(f"Total files: {len(urls)}")
    print(f"Files saved to: {download_dir}")

if __name__ == "__main__":
    main()


