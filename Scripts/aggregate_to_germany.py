import pandas as pd
import os
import glob
from pathlib import Path

def aggregate_by_germany_from_bundesland(input_folder, output_folder):
    """
    Aggregate all Bundesland files for one weather attribute into a single Germany-wide file
    
    Args:
        input_folder: Path to folder containing Bundesland CSV files
        output_folder: Path to output folder for the aggregated Germany file
    """
    
    print(f"Processing {os.path.basename(input_folder)}...")
    
    # Get all Bundesland CSV files
    pattern = os.path.join(input_folder, '*.csv')
    bundesland_files = glob.glob(pattern)
    
    if not bundesland_files:
        print(f"No files found in {input_folder}")
        return False
    
    print(f"Found {len(bundesland_files)} Bundesland files")
    
    # Combine all Bundesland data
    all_data = []
    all_stations = set()
    
    for file_path in bundesland_files:
        bundesland_name = os.path.basename(file_path).replace('.csv', '')
        print(f"  Reading {bundesland_name}...")
        
        try:
            df = pd.read_csv(file_path)
            all_data.append(df)
            
            # Collect all unique station IDs
            if 'stations' in df.columns:
                stations_str = df['stations'].iloc[0] if len(df) > 0 else ''
                stations_list = [s.strip() for s in stations_str.split(',')]
                all_stations.update(stations_list)
            
            print(f"    Added {len(df)} records from {bundesland_name}")
            
        except Exception as e:
            print(f"    Error reading {file_path}: {e}")
    
    if not all_data:
        print("No data to aggregate!")
        return False
    
    # Combine all dataframes
    print("\nCombining all data...")
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Add all_stations column
    combined_df['all_stations'] = ','.join(sorted(all_stations))
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Save aggregated Germany file
    output_file = os.path.join(output_folder, 'Germany_total.csv')
    combined_df.to_csv(output_file, index=False)
    
    print(f"Saved {len(combined_df)} total records to {output_file}")
    print(f"Total stations: {len(all_stations)}")
    print(f"Bundesland files processed: {len(bundesland_files)}")
    
    return True

def main():
    """
    Main function to process all weather attribute folders
    """
    
    # List of all weather attributes that were processed
    attributes = [
        'pressure',
        'soil_temperature',
        'sun',
        'visibility',
        'weather_phenomena',
        'wind',
        'wind_synop',
        'precipitation'
    ]
    
    base_path = r'D:\Bachelor abroad\Data'
    
    print("="*60)
    print("Aggregating all Bundesland data into Germany-wide files")
    print("="*60)
    
    for attribute in attributes:
        input_folder = os.path.join(base_path, f'{attribute}_by_bundesland')
        output_folder = os.path.join(base_path, f'{attribute}_germany')
        
        print(f"\n{'='*60}")
        print(f"Processing {attribute}")
        print(f"{'='*60}")
        
        success = aggregate_by_germany_from_bundesland(input_folder, output_folder)
        
        if success:
            print(f"✅ Successfully aggregated {attribute}")
        else:
            print(f"❌ Failed to aggregate {attribute}")
    
    print(f"\n{'='*60}")
    print("All processing complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
