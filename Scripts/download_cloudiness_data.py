#!/usr/bin/env python3
"""
Script to download all cloudiness data files from the HTML index page.
This script extracts all href URLs from the HTML file and downloads them.
"""

import re
import os
import requests
from urllib.parse import urljoin, urlparse
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

def extract_urls_from_html(html_file_path):
    """Extract all href URLs from the HTML file."""
    urls = []
    
    try:
        with open(html_file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            
        # Find all href attributes
        href_pattern = r'href="([^"]*)"'
        matches = re.findall(href_pattern, content)
        
        # Filter out relative links and keep only full URLs
        for match in matches:
            if match.startswith('http'):
                urls.append(match)
        
        print(f"Found {len(urls)} URLs to download")
        return urls
        
    except Exception as e:
        print(f"Error reading HTML file: {e}")
        return []

def download_file(url, download_dir="Cloudness_downloads"):
    """Download a single file from URL."""
    try:
        # Create download directory if it doesn't exist
        Path(download_dir).mkdir(exist_ok=True)
        
        # Parse URL to get filename
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        
        # Skip if it's a directory or empty filename
        if not filename or filename.endswith('/'):
            return None, f"Skipped directory or empty filename: {url}"
        
        file_path = os.path.join(download_dir, filename)
        
        # Skip if file already exists
        if os.path.exists(file_path):
            return filename, f"File already exists: {filename}"
        
        # Download the file
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Get file size for progress tracking
        total_size = int(response.headers.get('content-length', 0))
        
        with open(file_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
        
        return filename, f"Downloaded: {filename} ({downloaded} bytes)"
        
    except requests.exceptions.RequestException as e:
        return None, f"Error downloading {url}: {e}"
    except Exception as e:
        return None, f"Unexpected error for {url}: {e}"

def download_files_parallel(urls, max_workers=5):
    """Download files in parallel using ThreadPoolExecutor."""
    results = []
    successful_downloads = 0
    skipped_files = 0
    errors = 0
    
    print(f"Starting download of {len(urls)} files with {max_workers} workers...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_url = {executor.submit(download_file, url): url for url in urls}
        
        # Process completed downloads
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                filename, message = future.result()
                results.append((url, filename, message))
                
                if filename:
                    if "already exists" in message:
                        skipped_files += 1
                    else:
                        successful_downloads += 1
                else:
                    errors += 1
                
                print(f"[{len(results)}/{len(urls)}] {message}")
                
            except Exception as e:
                results.append((url, None, f"Exception: {e}"))
                errors += 1
                print(f"[{len(results)}/{len(urls)}] Error: {e}")
    
    return results, successful_downloads, skipped_files, errors

def main():
    """Main function to orchestrate the download process."""
    html_file = "Cloudness/Index of _climate_environment_CDC_observations_germany_climate_hourly_cloudiness_historical_.html"
    
    print("=== Cloudiness Data Downloader ===")
    print(f"Reading HTML file: {html_file}")
    
    # Extract URLs from HTML
    urls = extract_urls_from_html(html_file)
    
    if not urls:
        print("No URLs found in HTML file!")
        return
    
    print(f"\nFound {len(urls)} URLs to download")
    print("Starting downloads...\n")
    
    # Download files
    start_time = time.time()
    results, successful, skipped, errors = download_files_parallel(urls, max_workers=5)
    end_time = time.time()
    
    # Print summary
    print(f"\n=== Download Summary ===")
    print(f"Total files processed: {len(urls)}")
    print(f"Successfully downloaded: {successful}")
    print(f"Skipped (already exist): {skipped}")
    print(f"Errors: {errors}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    
    # Save results to file
    with open("download_results.txt", "w", encoding="utf-8") as f:
        f.write("Download Results\n")
        f.write("=" * 50 + "\n")
        for url, filename, message in results:
            f.write(f"URL: {url}\n")
            f.write(f"File: {filename}\n")
            f.write(f"Status: {message}\n")
            f.write("-" * 30 + "\n")
    
    print(f"\nDetailed results saved to: download_results.txt")
    print(f"Downloaded files are in: Cloudness_downloads/")

if __name__ == "__main__":
    main()
