#!/usr/bin/env python3
"""
Script to download all files from the HTML index page.
This script extracts all hyperlinks from the HTML file and downloads them.
"""

import re
import os
import requests
from urllib.parse import urljoin, urlparse
from pathlib import Path
import time

def extract_urls_from_html(html_file_path):
    """Extract all URLs from the HTML file."""
    urls = []
    
    with open(html_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Find all href attributes
    href_pattern = r'href="([^"]*)"'
    matches = re.findall(href_pattern, content)
    
    for match in matches:
        # Skip parent directory links and empty links
        if match and not match.endswith('/') and not match.startswith('#'):
            # Convert relative URLs to absolute URLs
            if match.startswith('http'):
                urls.append(match)
            else:
                # This is a relative URL, we need to construct the full URL
                base_url = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/cloud_type/historical/"
                full_url = urljoin(base_url, match)
                urls.append(full_url)
    
    return urls

def download_file(url, output_dir):
    """Download a single file from URL to output directory."""
    try:
        # Parse the URL to get the filename
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        
        if not filename:
            print(f"Warning: Could not extract filename from {url}")
            return False
        
        output_path = os.path.join(output_dir, filename)
        
        # Skip if file already exists
        if os.path.exists(output_path):
            print(f"File {filename} already exists, skipping...")
            return True
        
        print(f"Downloading {filename}...")
        
        # Download the file
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Write the file
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Successfully downloaded {filename}")
        return True
        
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return False

def main():
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    html_file = script_dir / "Index of _climate_environment_CDC_observations_germany_climate_hourly_cloud_type_historical_.html"
    
    if not html_file.exists():
        print(f"HTML file not found: {html_file}")
        return
    
    print("Extracting URLs from HTML file...")
    urls = extract_urls_from_html(html_file)
    
    print(f"Found {len(urls)} URLs to download")
    
    # Create output directory (same as script directory)
    output_dir = script_dir
    
    # Download each file
    successful_downloads = 0
    failed_downloads = 0
    
    for i, url in enumerate(urls, 1):
        print(f"\nProgress: {i}/{len(urls)}")
        if download_file(url, output_dir):
            successful_downloads += 1
        else:
            failed_downloads += 1
        
        # Add a small delay to be respectful to the server
        time.sleep(0.5)
    
    print(f"\nDownload complete!")
    print(f"Successfully downloaded: {successful_downloads}")
    print(f"Failed downloads: {failed_downloads}")

if __name__ == "__main__":
    main()
