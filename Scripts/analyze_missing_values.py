#!/usr/bin/env python3
"""
Analyze missing values (NaN, empty strings, whitespace) in CSV files.
Calculates percentages of missing values for each attribute in each file.
Processes all CSV files recursively in the Data directory.
"""

import pandas as pd
import os
import glob
from pathlib import Path

def count_missing_values(df):
    """
    Count missing values in a DataFrame.
    Missing values include: NaN, None, empty strings, and whitespace-only strings.
    
    Returns:
        dict: Dictionary with column names as keys and missing value counts as values
    """
    missing_counts = {}
    
    for col in df.columns:
        # Count NaN and None values
        nan_count = df[col].isna().sum()
        
        # Count empty strings (only for object/string columns)
        if df[col].dtype == 'object':
            empty_string_count = (df[col].astype(str).str.strip() == '').sum()
            # Subtract empty strings that are already NaN to avoid double counting
            empty_string_count = empty_string_count - (df[col].isna() & (df[col].astype(str).str.strip() == '')).sum()
            total_missing = nan_count + empty_string_count
        else:
            total_missing = nan_count
        
        missing_counts[col] = {
            'missing_count': total_missing,
            'total_rows': len(df),
            'missing_percentage': (total_missing / len(df)) * 100 if len(df) > 0 else 0
        }
    
    return missing_counts

def analyze_csv_file(file_path, base_path):
    """
    Analyze a single CSV file for missing values.
    
    Args:
        file_path: Full path to the CSV file
        base_path: Base directory path for relative path calculation
    
    Returns:
        dict: Results dictionary with file name and missing value statistics
    """
    try:
        df = pd.read_csv(file_path, low_memory=False)
        file_name = os.path.basename(file_path)
        # Get relative folder path from base directory
        rel_folder = os.path.relpath(os.path.dirname(file_path), base_path)
        
        missing_stats = count_missing_values(df)
        
        return {
            'file': file_name,
            'folder': rel_folder,
            'full_path': file_path,
            'total_rows': len(df),
            'columns': list(df.columns),
            'missing_stats': missing_stats
        }
    except Exception as e:
        return {
            'file': os.path.basename(file_path),
            'folder': os.path.relpath(os.path.dirname(file_path), base_path),
            'full_path': file_path,
            'error': str(e)
        }

def find_all_csv_files(base_directory):
    """
    Recursively find all CSV files in a directory tree.
    Excludes summary files (starting with '_').
    
    Args:
        base_directory: Root directory to search
    
    Returns:
        list: List of full paths to CSV files
    """
    csv_files = []
    for root, dirs, files in os.walk(base_directory):
        for file in files:
            if file.endswith('.csv') and not file.startswith('_'):
                csv_files.append(os.path.join(root, file))
    return sorted(csv_files)

def analyze_all_data_folders(base_directory, output_file=None, verbose=True):
    """
    Analyze all CSV files in all subdirectories of the base directory.
    
    Args:
        base_directory: Root directory to search (e.g., "Data")
        output_file: Optional path to save results as CSV
        verbose: Whether to print detailed progress
    
    Returns:
        pd.DataFrame: DataFrame with all missing value statistics
    """
    # Find all CSV files
    csv_files = find_all_csv_files(base_directory)
    
    if verbose:
        print(f"Found {len(csv_files)} CSV files to analyze across all folders\n")
        print("=" * 120)
    
    all_results = []
    files_processed = 0
    files_with_errors = 0
    
    for file_path in csv_files:
        files_processed += 1
        rel_folder = os.path.relpath(os.path.dirname(file_path), base_directory)
        
        if verbose and files_processed % 10 == 0:
            print(f"Processing file {files_processed}/{len(csv_files)}: {rel_folder}/{os.path.basename(file_path)}")
        
        result = analyze_csv_file(file_path, base_directory)
        
        if 'error' in result:
            files_with_errors += 1
            if verbose:
                print(f"ERROR processing {file_path}: {result['error']}")
            # Still add error to results
            all_results.append({
                'Folder': result['folder'],
                'File': result['file'],
                'Full_Path': result['full_path'],
                'Column': 'ERROR',
                'Total_Rows': 0,
                'Missing_Count': 0,
                'Missing_Percentage': 0.0,
                'Error': result['error']
            })
            continue
        
        # Add results for each column
        for col, stats in result['missing_stats'].items():
            all_results.append({
                'Folder': result['folder'],
                'File': result['file'],
                'Full_Path': result['full_path'],
                'Column': col,
                'Total_Rows': stats['total_rows'],
                'Missing_Count': stats['missing_count'],
                'Missing_Percentage': round(stats['missing_percentage'], 2)
            })
    
    # Create summary DataFrame
    if all_results:
        summary_df = pd.DataFrame(all_results)
        
        if output_file:
            summary_df.to_csv(output_file, index=False)
            if verbose:
                print(f"\n\nResults saved to: {output_file}")
        
        # Print summary statistics
        if verbose:
            print("\n\n" + "=" * 120)
            print("SUMMARY")
            print("=" * 120)
            
            print(f"\nTotal files processed: {files_processed}")
            print(f"Files with errors: {files_with_errors}")
            print(f"Files successfully analyzed: {files_processed - files_with_errors}")
            
            # Remove error rows for statistics
            valid_df = summary_df[summary_df['Column'] != 'ERROR'].copy()
            
            if len(valid_df) > 0:
                # Unique folders
                unique_folders = valid_df['Folder'].unique()
                print(f"Unique folders analyzed: {len(unique_folders)}")
                
                # Unique files
                unique_files = valid_df['File'].unique()
                print(f"Unique files analyzed: {len(unique_files)}")
                
                # Files with any missing values
                files_with_missing = valid_df[valid_df['Missing_Count'] > 0]['File'].unique()
                print(f"Files with missing values: {len(files_with_missing)} out of {len(unique_files)}")
                
                # Columns with missing values
                cols_with_missing = valid_df[valid_df['Missing_Count'] > 0]['Column'].unique()
                print(f"Columns with missing values: {len(cols_with_missing)}")
                
                # Files and columns summary by folder
                print("\n" + "=" * 120)
                print("SUMMARY BY FOLDER")
                print("=" * 120)
                folder_summary = valid_df.groupby('Folder').agg({
                    'File': 'nunique',
                    'Column': 'nunique',
                    'Missing_Count': 'sum'
                }).reset_index()
                folder_summary.columns = ['Folder', 'Files', 'Unique_Columns', 'Total_Missing_Values']
                folder_summary = folder_summary.sort_values('Total_Missing_Values', ascending=False)
                print(folder_summary.to_string(index=False))
                
                # Columns with highest missing percentages
                print("\n" + "=" * 120)
                print("TOP 20 COLUMNS WITH HIGHEST MISSING PERCENTAGES")
                print("=" * 120)
                top_missing = valid_df.nlargest(20, 'Missing_Percentage')[['Folder', 'File', 'Column', 'Missing_Percentage', 'Missing_Count', 'Total_Rows']]
                print(top_missing.to_string(index=False))
        
        return summary_df
    else:
        if verbose:
            print("\nNo results to report.")
        return None

def main():
    # Base directory path - analyze all folders in Data directory
    base_directory = r"Data"
    
    # Check if directory exists
    if not os.path.exists(base_directory):
        print(f"Directory not found: {base_directory}")
        print("Please specify the correct path to the Data directory.")
        return
    
    # Output file for results
    output_file = "Data/missing_values_analysis_all_folders.csv"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print("Starting comprehensive missing values analysis...")
    print(f"Searching in: {os.path.abspath(base_directory)}")
    print(f"Results will be saved to: {os.path.abspath(output_file)}")
    print("\n")
    
    # Analyze all files
    summary_df = analyze_all_data_folders(base_directory, output_file, verbose=True)
    
    print("\n\nAnalysis complete!")
    if summary_df is not None:
        print(f"Total records in output file: {len(summary_df)}")

if __name__ == "__main__":
    main()
