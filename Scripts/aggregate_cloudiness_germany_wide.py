#!/usr/bin/env python3
"""
Aggregate all cloudiness data across Germany into a single dataset.
Creates a Germany_Aggregated folder with country-wide cloudiness statistics.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

def aggregate_cloudiness_germany_wide():
    """Main function to aggregate all cloudiness data for Germany."""
    
    print("🇩🇪 Starting Germany-wide cloudiness aggregation...")
    
    # Get all cloudiness files
    cloudiness_dir = "Cloudness_downloads"
    csv_files = []
    
    for filename in os.listdir(cloudiness_dir):
        if filename.endswith('.csv') and filename != 'N_Stundenwerte_Beschreibung_Stationen.csv':
            # Extract station ID from filename (remove .csv extension)
            station_id = filename.replace('.csv', '')
            csv_files.append((station_id, os.path.join(cloudiness_dir, filename)))
    
    print(f"📁 Found {len(csv_files)} cloudiness files")
    
    # Create output directory
    output_dir = "Cloudness_Germany_Aggregated"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and combine all data
    all_data = []
    station_ids = []
    
    print("📊 Loading and combining all cloudiness data...")
    
    for station_id, file_path in csv_files:
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
        print("❌ No valid data found")
        return
    
    # Combine all data
    print("🔄 Combining data from all stations...")
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Convert MESS_DATUM to datetime
    combined_df['MESS_DATUM'] = pd.to_datetime(combined_df['MESS_DATUM'])
    
    print(f"📈 Total records: {len(combined_df):,}")
    print(f"🏢 Total stations: {len(station_ids)}")
    
    # Group by datetime and aggregate
    print("📊 Aggregating data by hour...")
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
    output_filename = "Germany_cloudiness_aggregated.csv"
    output_path = os.path.join(output_dir, output_filename)
    aggregated_df.to_csv(output_path, index=False)
    
    # Calculate statistics
    total_rows = len(combined_df)
    aggregated_rows = len(aggregated_df)
    unique_stations = len(station_ids)
    avg_stations_per_hour = aggregated_df['STATION_COUNT'].mean()
    avg_cloudiness_2024 = aggregated_df['V_N_mean'].mean()
    
    # Create summary statistics
    summary_stats = {
        'Metric': [
            'Total Original Records',
            'Aggregated Records', 
            'Unique Stations',
            'Avg Stations Per Hour',
            'Avg Cloudiness 2024',
            'Date Range Start',
            'Date Range End',
            'Total Hours Covered'
        ],
        'Value': [
            f"{total_rows:,}",
            f"{aggregated_rows:,}",
            f"{unique_stations}",
            f"{avg_stations_per_hour:.1f}",
            f"{avg_cloudiness_2024:.2f}",
            str(aggregated_df['MESS_DATUM'].min()),
            str(aggregated_df['MESS_DATUM'].max()),
            f"{len(aggregated_df)} hours"
        ]
    }
    
    summary_df = pd.DataFrame(summary_stats)
    summary_path = os.path.join(output_dir, "Germany_cloudiness_statistics.csv")
    summary_df.to_csv(summary_path, index=False)
    
    # Create a text summary file
    summary_text = f"""Germany-Wide Cloudiness Data Aggregation Summary
====================================================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Data Overview:
- Total Original Records: {total_rows:,}
- Aggregated Records: {aggregated_rows:,}
- Unique Stations: {unique_stations}
- Average Stations Per Hour: {avg_stations_per_hour:.1f}
- Average Cloudiness 2024: {avg_cloudiness_2024:.2f}

Date Range:
- Start: {aggregated_df['MESS_DATUM'].min()}
- End: {aggregated_df['MESS_DATUM'].max()}
- Total Hours: {len(aggregated_df)} hours

Data Quality:
- All stations contributing to each hour: {len(aggregated_df[aggregated_df['STATION_COUNT'] == unique_stations])} hours
- Minimum stations per hour: {aggregated_df['STATION_COUNT'].min()}
- Maximum stations per hour: {aggregated_df['STATION_COUNT'].max()}

Files Created:
- Germany_cloudiness_aggregated.csv: Main aggregated dataset
- Germany_cloudiness_statistics.csv: Summary statistics
- Germany_cloudiness_statistics.txt: This summary file

Data Structure:
- MESS_DATUM: Date and time (2024 data only)
- V_N_mean: Average cloudiness value across all stations
- QN_8_mean: Average quality indicator
- V_N_I_mean: Most common cloudiness indicator
- STATION_COUNT: Number of stations contributing to each hour
- STATIONS_CONTRIBUTING: List of station IDs used for each hour

This dataset provides a comprehensive view of cloudiness patterns across all of Germany for 2024.
"""
    
    summary_text_path = os.path.join(output_dir, "Germany_cloudiness_statistics.txt")
    with open(summary_text_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    
    print(f"\n📊 Summary saved to {summary_path}")
    print(f"📄 Text summary saved to {summary_text_path}")
    print(f"📁 All files saved to {output_dir}/")
    
    # Print summary
    print("\n📈 Germany-Wide Aggregation Summary:")
    print(f"✅ Total records: {total_rows:,} → {aggregated_rows:,}")
    print(f"🏢 Stations: {unique_stations}")
    print(f"📊 Avg stations/hour: {avg_stations_per_hour:.1f}")
    print(f"☁️ Avg cloudiness: {avg_cloudiness_2024:.2f}")
    print(f"📅 Date range: {aggregated_df['MESS_DATUM'].min()} to {aggregated_df['MESS_DATUM'].max()}")
    
    return aggregated_df

if __name__ == "__main__":
    aggregate_cloudiness_germany_wide()
