#!/usr/bin/env python3
"""
Script to download all extreme wind data files from the DWD (German Weather Service) website.
This script parses the HTML index file and downloads all ZIP files to the extreme_wind directory.
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
    
    with open(html_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all href attributes that point to ZIP files
    pattern = r'href="(https://opendata\.dwd\.de[^"]*\.zip)"'
    matches = re.findall(pattern, content)
    
    return matches

def download_file(url, output_dir):
    """Download a single file from URL to output directory."""
    try:
        # Extract filename from URL
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        filepath = os.path.join(output_dir, filename)
        
        # Skip if file already exists
        if os.path.exists(filepath):
            print(f"✓ Skipping {filename} (already exists)")
            return True
        
        print(f"📥 Downloading {filename}...")
        
        # Download the file
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Write file in chunks
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✓ Downloaded {filename}")
        return True
        
    except Exception as e:
        print(f"✗ Error downloading {url}: {e}")
        return False

def main():
    """Main function to download all extreme wind data files."""
    # Set up paths
    script_dir = Path(__file__).parent
    html_file = script_dir / "extreme_wind" / "Index of _climate_environment_CDC_observations_germany_climate_hourly_extreme_wind_historical_.html"
    output_dir = script_dir / "extreme_wind"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    print("🔍 Extracting download URLs from HTML file...")
    urls = extract_download_urls(html_file)
    print(f"Found {len(urls)} files to download")
    
    if not urls:
        print("❌ No URLs found in HTML file")
        return
    
    # Download files
    successful_downloads = 0
    failed_downloads = 0
    
    print(f"\n📦 Starting download of {len(urls)} files...")
    print("=" * 60)
    
    for i, url in enumerate(urls, 1):
        print(f"[{i}/{len(urls)}] ", end="")
        
        if download_file(url, output_dir):
            successful_downloads += 1
        else:
            failed_downloads += 1
        
        # Add a small delay to be respectful to the server
        time.sleep(0.5)
    
    print("\n" + "=" * 60)
    print(f"📊 Download Summary:")
    print(f"   ✓ Successful: {successful_downloads}")
    print(f"   ✗ Failed: {failed_downloads}")
    print(f"   📁 Files saved to: {output_dir}")
    
    if failed_downloads > 0:
        print(f"\n⚠️  {failed_downloads} files failed to download. You may want to retry them.")

if __name__ == "__main__":
    main()


