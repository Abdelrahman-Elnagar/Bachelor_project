#!/usr/bin/env python3
"""
Script to remove all downloaded extreme wind data files and re-download them.
"""

import os
import re
import requests
import time
from urllib.parse import urlparse
from pathlib import Path
import glob

def clean_downloaded_files(output_dir):
    """Remove all ZIP files from the output directory."""
    print("🧹 Cleaning downloaded files...")
    
    # Find all ZIP files in the directory
    zip_files = glob.glob(os.path.join(output_dir, "*.zip"))
    
    if not zip_files:
        print("✓ No ZIP files found to remove")
        return
    
    print(f"Found {len(zip_files)} ZIP files to remove")
    
    removed_count = 0
    for zip_file in zip_files:
        try:
            os.remove(zip_file)
            removed_count += 1
            print(f"✓ Removed {os.path.basename(zip_file)}")
        except Exception as e:
            print(f"✗ Error removing {zip_file}: {e}")
    
    print(f"✓ Removed {removed_count} files")

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
    """Main function to clean and re-download all extreme wind data files."""
    # Set up paths
    script_dir = Path(__file__).parent
    html_file = script_dir / "extreme_wind" / "Index of _climate_environment_CDC_observations_germany_climate_hourly_extreme_wind_historical_.html"
    output_dir = script_dir / "extreme_wind"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Step 1: Clean existing files
    clean_downloaded_files(output_dir)
    
    print("\n" + "=" * 60)
    
    # Step 2: Extract URLs
    print("🔍 Extracting download URLs from HTML file...")
    urls = extract_download_urls(html_file)
    print(f"Found {len(urls)} files to download")
    
    if not urls:
        print("❌ No URLs found in HTML file")
        return
    
    # Step 3: Download files
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

