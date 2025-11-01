#!/usr/bin/env python3
"""
Aggregate cloudiness data by Bundesland (German states).
Creates a Cloudness_Bundesland_Aggregated folder with files for each state.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

def load_regions_mapping():
    """Load the regions.csv file to get station-to-Bundesland mapping."""
    regions_df = pd.read_csv('regions.csv')
    
    # Create mapping from station_id to Bundesland
    station_to_bundesland = {}
    for _, row in regions_df.iterrows():
        # Try both with and without leading zeros
        station_id_with_zeros = str(row['Stations_id']).zfill(5)  # 5-digit format
        station_id_no_zeros = str(int(row['Stations_id']))  # No leading zeros
        bundesland = row['Bundesland']
        station_to_bundesland[station_id_with_zeros] = bundesland
        station_to_bundesland[station_id_no_zeros] = bundesland
    
    return station_to_bundesland

def get_available_cloudiness_files():
    """Get list of available cloudiness CSV files."""
    cloudiness_dir = "Cloudness_downloads"
    csv_files = []
    
    for filename in os.listdir(cloudiness_dir):
        if filename.endswith('.csv') and filename != 'N_Stundenwerte_Beschreibung_Stationen.csv':
            # Extract station ID from filename (remove .csv extension)
            station_id = filename.replace('.csv', '')
            csv_files.append((station_id, os.path.join(cloudiness_dir, filename)))
    
    return csv_files

def aggregate_cloudiness_by_bundesland():
    """Main function to aggregate cloudiness data by Bundesland."""
    
    print("🌤️ Starting cloudiness aggregation by Bundesland...")
    
    # Load station-to-Bundesland mapping
    print("📋 Loading regions mapping...")
    station_to_bundesland = load_regions_mapping()
    
    # Get available cloudiness files
    print("📁 Scanning cloudiness files...")
    cloudiness_files = get_available_cloudiness_files()
    print(f"Found {len(cloudiness_files)} cloudiness files")
    
    # Group files by Bundesland
    bundesland_files = {}
    for station_id, file_path in cloudiness_files:
        if station_id in station_to_bundesland:
            bundesland = station_to_bundesland[station_id]
            if bundesland not in bundesland_files:
                bundesland_files[bundesland] = []
            bundesland_files[bundesland].append((station_id, file_path))
        else:
            print(f"⚠️ Station {station_id} not found in regions mapping")
    
    print(f"📊 Found data for {len(bundesland_files)} Bundesländer")
    
    # Create output directory
    output_dir = "Cloudness_Bundesland_Aggregated"
    os.makedirs(output_dir, exist_ok=True)
    
    # Statistics for summary
    summary_stats = []
    
    # Process each Bundesland
    for bundesland, files in bundesland_files.items():
        print(f"\n🏛️ Processing {bundesland} ({len(files)} stations)...")
        
        all_data = []
        station_ids = []
        
        # Load and combine data from all stations in this Bundesland
        for station_id, file_path in files:
            try:
                df = pd.read_csv(file_path)
                
                # Add station ID column
                df['STATION_ID'] = station_id
                all_data.append(df)
                station_ids.append(station_id)
                
            except Exception as e:
                print(f"⚠️ Error reading {file_path}: {e}")
                continue
        
        if not all_data:
            print(f"❌ No valid data for {bundesland}")
            continue
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Convert MESS_DATUM to datetime
        combined_df['MESS_DATUM'] = pd.to_datetime(combined_df['MESS_DATUM'])
        
        # Group by datetime and aggregate
        aggregated_data = []
        
        for datetime_val, group in combined_df.groupby('MESS_DATUM'):
            # Calculate mean values for cloudiness metrics
            # Handle the space in column name " V_N"
            v_n_col = 'V_N' if 'V_N' in group.columns else ' V_N' if ' V_N' in group.columns else None
            
            row_data = {
                'MESS_DATUM': datetime_val,
                'V_N_mean': group[v_n_col].mean() if v_n_col is not None else np.nan,
                'QN_8_mean': group['QN_8'].mean() if 'QN_8' in group.columns else np.nan,
                'V_N_I_mean': group['V_N_I'].mode().iloc[0] if 'V_N_I' in group.columns and not group['V_N_I'].mode().empty else np.nan,
                'STATION_COUNT': len(group),
                'STATIONS_CONTRIBUTING': ','.join(sorted(group['STATION_ID'].unique()))
            }
            aggregated_data.append(row_data)
        
        # Create aggregated DataFrame
        aggregated_df = pd.DataFrame(aggregated_data)
        aggregated_df = aggregated_df.sort_values('MESS_DATUM').reset_index(drop=True)
        
        # Save to CSV
        output_filename = f"{bundesland}_aggregated.csv"
        output_path = os.path.join(output_dir, output_filename)
        aggregated_df.to_csv(output_path, index=False)
        
        # Calculate statistics
        total_rows = len(combined_df)
        aggregated_rows = len(aggregated_df)
        unique_stations = len(station_ids)
        avg_stations_per_hour = aggregated_df['STATION_COUNT'].mean()
        avg_cloudiness_2024 = aggregated_df['V_N_mean'].mean()
        
        summary_stats.append({
            'Bundesland': bundesland,
            'Original_Rows': total_rows,
            'Aggregated_Rows': aggregated_rows,
            'Unique_Stations': unique_stations,
            'Avg_Stations_Per_Hour': round(avg_stations_per_hour, 1),
            'Avg_Cloudiness_2024': round(avg_cloudiness_2024, 2)
        })
        
        print(f"✅ {bundesland}: {total_rows} → {aggregated_rows} rows, {unique_stations} stations")
    
    # Create summary file
    summary_df = pd.DataFrame(summary_stats)
    summary_df = summary_df.sort_values('Unique_Stations', ascending=False)
    summary_path = os.path.join(output_dir, '_AGGREGATION_SUMMARY.csv')
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\n📊 Summary saved to {summary_path}")
    print(f"📁 All files saved to {output_dir}/")
    
    # Print summary
    print("\n📈 Aggregation Summary:")
    print(summary_df.to_string(index=False))
    
    return summary_df

if __name__ == "__main__":
    aggregate_cloudiness_by_bundesland()
