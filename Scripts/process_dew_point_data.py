import pandas as pd
import os
import glob
from pathlib import Path

def process_dew_point_data():
    """
    Process dew point data and aggregate by Bundesland
    """
    
    # Read regions.csv
    print("Reading regions.csv...")
    regions_df = pd.read_csv(r'd:\Bachelor abroad\regions.csv')
    
    # Create mapping from station_id (with leading zeros) to Bundesland
    # Remove leading zeros from station_id for matching with dew point files
    regions_df['station_id_numeric'] = regions_df['Stations_id'].astype(str).str.lstrip('0')
    regions_df['station_id_numeric'] = regions_df['station_id_numeric'].replace('', '0')  # Handle case where all zeros are removed
    
    # Create mapping dictionary
    station_to_bundesland = dict(zip(regions_df['station_id_numeric'], regions_df['Bundesland']))
    
    print(f"Found {len(station_to_bundesland)} stations in regions.csv")
    
    # Get all dew point files
    dew_point_dir = r'd:\Bachelor abroad\dew_point'
    dew_point_files = glob.glob(os.path.join(dew_point_dir, '*.csv'))
    
    # Filter out the description file
    dew_point_files = [f for f in dew_point_files if not f.endswith('TD_Stundenwerte_Beschreibung_Stationen.csv')]
    
    print(f"Found {len(dew_point_files)} dew point files")
    
    # Group files by Bundesland
    bundesland_data = {}
    bundesland_stations = {}
    
    for file_path in dew_point_files:
        # Extract station ID from filename
        filename = os.path.basename(file_path)
        station_id = filename.replace('.csv', '')
        
        # Find corresponding Bundesland
        bundesland = station_to_bundesland.get(station_id)
        
        if bundesland:
            print(f"Processing station {station_id} -> {bundesland}")
            
            # Read dew point data
            try:
                dew_data = pd.read_csv(file_path)
                
                # Initialize if first station for this Bundesland
                if bundesland not in bundesland_data:
                    bundesland_data[bundesland] = []
                    bundesland_stations[bundesland] = []
                
                # Add station ID to the data
                dew_data['station_id'] = station_id
                
                # Store data and station info
                bundesland_data[bundesland].append(dew_data)
                bundesland_stations[bundesland].append(station_id)
                
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        else:
            print(f"No Bundesland found for station {station_id}")
    
    # Create output directory
    output_dir = r'd:\Bachelor abroad\dew_point_by_bundesland'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each Bundesland
    for bundesland, dataframes in bundesland_data.items():
        print(f"\nProcessing {bundesland} with {len(dataframes)} stations...")
        
        # Combine all dataframes for this Bundesland
        combined_df = pd.concat(dataframes, ignore_index=True)
        
        # Add stations column with all station IDs used
        stations_list = bundesland_stations[bundesland]
        combined_df['stations'] = ','.join(stations_list)
        
        # Save to file (sanitize filename)
        safe_filename = bundesland.replace(' ', '_').replace('/', '_').replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'ue').replace('ß', 'ss')
        output_file = os.path.join(output_dir, f'{safe_filename}.csv')
        
        combined_df.to_csv(output_file, index=False)
        print(f"Saved {len(combined_df)} records to {output_file}")
        print(f"Stations included: {', '.join(stations_list)}")
    
    print(f"\nProcessing complete! Files saved to {output_dir}")
    
    # Print summary
    print("\nSummary:")
    for bundesland in bundesland_data.keys():
        print(f"{bundesland}: {len(bundesland_data[bundesland])} stations, {len(bundesland_stations[bundesland])} station IDs")

if __name__ == "__main__":
    process_dew_point_data()
