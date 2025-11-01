import pandas as pd
import os
import glob
from pathlib import Path
import argparse
import sys

def process_weather_data_by_bundesland(input_folder, output_folder, regions_file, attribute_name="weather_data"):
    """
    Process weather data files and aggregate by Bundesland
    
    Args:
        input_folder (str): Path to folder containing CSV files named by station ID
        output_folder (str): Path to output folder for Bundesland files
        regions_file (str): Path to regions.csv file
        attribute_name (str): Name of the attribute being processed (for logging)
    """
    
    print(f"Processing {attribute_name} data...")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Regions file: {regions_file}")
    
    # Check if input folder exists
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist!")
        return False
    
    # Check if regions file exists
    if not os.path.exists(regions_file):
        print(f"Error: Regions file '{regions_file}' does not exist!")
        return False
    
    # Read regions.csv
    print("Reading regions.csv...")
    try:
        regions_df = pd.read_csv(regions_file)
    except Exception as e:
        print(f"Error reading regions file: {e}")
        return False
    
    # Create mapping from station_id (with leading zeros) to Bundesland
    # Remove leading zeros from station_id for matching with data files
    regions_df['station_id_numeric'] = regions_df['Stations_id'].astype(str).str.lstrip('0')
    regions_df['station_id_numeric'] = regions_df['station_id_numeric'].replace('', '0')  # Handle case where all zeros are removed
    
    # Create mapping dictionary
    station_to_bundesland = dict(zip(regions_df['station_id_numeric'], regions_df['Bundesland']))
    
    print(f"Found {len(station_to_bundesland)} stations in regions.csv")
    
    # Get all CSV files from input folder
    csv_pattern = os.path.join(input_folder, '*.csv')
    data_files = glob.glob(csv_pattern)
    
    # Filter out description files (common patterns)
    description_patterns = ['beschreibung', 'description', 'info', 'metadata']
    data_files = [f for f in data_files if not any(pattern in os.path.basename(f).lower() for pattern in description_patterns)]
    
    print(f"Found {len(data_files)} data files")
    
    if len(data_files) == 0:
        print("No data files found!")
        return False
    
    # Group files by Bundesland
    bundesland_data = {}
    bundesland_stations = {}
    unmatched_stations = []
    
    for file_path in data_files:
        # Extract station ID from filename
        filename = os.path.basename(file_path)
        station_id = filename.replace('.csv', '')
        
        # Find corresponding Bundesland
        bundesland = station_to_bundesland.get(station_id)
        
        if bundesland:
            print(f"Processing station {station_id} -> {bundesland}")
            
            # Read data file
            try:
                data_df = pd.read_csv(file_path)
                
                # Initialize if first station for this Bundesland
                if bundesland not in bundesland_data:
                    bundesland_data[bundesland] = []
                    bundesland_stations[bundesland] = []
                
                # Add station ID to the data
                data_df['station_id'] = station_id
                
                # Store data and station info
                bundesland_data[bundesland].append(data_df)
                bundesland_stations[bundesland].append(station_id)
                
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        else:
            unmatched_stations.append(station_id)
            print(f"No Bundesland found for station {station_id}")
    
    if unmatched_stations:
        print(f"\nUnmatched stations ({len(unmatched_stations)}): {', '.join(unmatched_stations[:10])}{'...' if len(unmatched_stations) > 10 else ''}")
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Process each Bundesland
    total_records = 0
    for bundesland, dataframes in bundesland_data.items():
        print(f"\nProcessing {bundesland} with {len(dataframes)} stations...")
        
        # Combine all dataframes for this Bundesland
        combined_df = pd.concat(dataframes, ignore_index=True)
        
        # Add stations column with all station IDs used
        stations_list = bundesland_stations[bundesland]
        combined_df['stations'] = ','.join(stations_list)
        
        # Save to file (sanitize filename)
        safe_filename = bundesland.replace(' ', '_').replace('/', '_').replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'ue').replace('ß', 'ss')
        output_file = os.path.join(output_folder, f'{safe_filename}.csv')
        
        combined_df.to_csv(output_file, index=False)
        print(f"Saved {len(combined_df)} records to {output_file}")
        print(f"Stations included: {', '.join(stations_list)}")
        
        total_records += len(combined_df)
    
    print(f"\nProcessing complete! Files saved to {output_folder}")
    print(f"Total records processed: {total_records}")
    
    # Print summary
    print("\nSummary:")
    for bundesland in bundesland_data.keys():
        print(f"{bundesland}: {len(bundesland_data[bundesland])} stations, {len(bundesland_stations[bundesland])} station IDs")
    
    return True

def main():
    """
    Main function with command line argument parsing
    """
    parser = argparse.ArgumentParser(description='Process weather data files and aggregate by Bundesland')
    parser.add_argument('input_folder', help='Path to folder containing CSV files named by station ID')
    parser.add_argument('output_folder', help='Path to output folder for Bundesland files')
    parser.add_argument('--regions', default='regions.csv', help='Path to regions.csv file (default: regions.csv)')
    parser.add_argument('--attribute', default='weather_data', help='Name of the attribute being processed (default: weather_data)')
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    input_folder = os.path.abspath(args.input_folder)
    output_folder = os.path.abspath(args.output_folder)
    regions_file = os.path.abspath(args.regions)
    
    success = process_weather_data_by_bundesland(
        input_folder=input_folder,
        output_folder=output_folder,
        regions_file=regions_file,
        attribute_name=args.attribute
    )
    
    if success:
        print("\n✅ Processing completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Processing failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
