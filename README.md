# Weather Data Processing by Bundesland

This script processes weather data files and aggregates them by German Bundesland (federal states). It's designed to work with any weather attribute (temperature, humidity, precipitation, dew point, etc.).

## Files

- `process_weather_by_bundesland.py` - Main processing script
- `usage_examples.py` - Usage examples and documentation
- `regions.csv` - Reference file with station IDs and Bundesland mapping

## Usage

### Basic Syntax
```bash
python process_weather_by_bundesland.py INPUT_FOLDER OUTPUT_FOLDER [OPTIONS]
```

### Required Arguments
- `INPUT_FOLDER` - Path to folder containing CSV files named by station ID
- `OUTPUT_FOLDER` - Path to output folder for Bundesland files

### Optional Arguments
- `--regions REGIONS_FILE` - Path to regions.csv file (default: regions.csv)
- `--attribute ATTRIBUTE_NAME` - Name of the attribute being processed (default: weather_data)

## Examples

### 1. Process Dew Point Data
```bash
python process_weather_by_bundesland.py dew_point dew_point_by_bundesland --attribute dew_point
```

### 2. Process Temperature Data
```bash
python process_weather_by_bundesland.py temperature_data temperature_by_bundesland --attribute temperature
```

### 3. Process Humidity Data with Custom Regions File
```bash
python process_weather_by_bundesland.py humidity_data humidity_by_bundesland --regions custom_regions.csv --attribute humidity
```

### 4. Process Precipitation Data
```bash
python process_weather_by_bundesland.py precipitation_data precipitation_by_bundesland --attribute precipitation
```

## Input Requirements

### Folder Structure
Your input folder should contain:
- CSV files named by station ID (e.g., `71.csv`, `257.csv`, `755.csv`)
- Each CSV should have weather data with a `STATIONS_ID` column
- Files should NOT include description/metadata files (automatically filtered)

### Example Input File Structure
```csv
STATIONS_ID,MESS_DATUM,QN_8,TT,TD
71,2024-01-01 00:00:00,3,3.0,2.0
71,2024-01-01 01:00:00,3,2.8,2.0
71,2024-01-01 02:00:00,3,2.7,1.9
...
```

## Output

### Folder Structure
The output folder will contain:
- One CSV file per Bundesland (e.g., `Bayern.csv`, `Baden-Wuerttemberg.csv`)
- Each file contains aggregated data from all stations in that Bundesland
- Additional columns: `station_id` and `stations`

### Example Output File Structure
```csv
STATIONS_ID,MESS_DATUM,QN_8,TT,TD,station_id,stations
71,2024-01-01 00:00:00,3,3.0,2.0,71,"71,257,755"
71,2024-01-01 01:00:00,3,2.8,2.0,71,"71,257,755"
257,2024-01-01 00:00:00,3,4.2,3.1,257,"71,257,755"
...
```

### Output Files by Bundesland
- `Baden-Wuerttemberg.csv`
- `Bayern.csv`
- `Berlin.csv`
- `Brandenburg.csv`
- `Bremen.csv`
- `Hamburg.csv`
- `Hessen.csv`
- `Mecklenburg-Vorpommern.csv`
- `Niedersachsen.csv`
- `Nordrhein-Westfalen.csv`
- `Rheinland-Pfalz.csv`
- `Saarland.csv`
- `Sachsen-Anhalt.csv`
- `Sachsen.csv`
- `Schleswig-Holstein.csv`
- `Thueringen.csv`

## Features

- **Flexible Input**: Works with any weather attribute data
- **Automatic Mapping**: Maps station IDs to Bundesland using regions.csv
- **Error Handling**: Gracefully handles missing files and invalid data
- **Progress Tracking**: Shows processing progress and statistics
- **File Filtering**: Automatically excludes description/metadata files
- **Sanitized Output**: Creates safe filenames for Bundesland files

## Station ID Mapping

The script handles the mapping between:
- Station IDs in `regions.csv` (with leading zeros): `00071`, `00257`, `00755`
- Station IDs in data files (without leading zeros): `71.csv`, `257.csv`, `755.csv`

## Error Handling

- Missing input folders or files are reported
- Invalid CSV files are skipped with error messages
- Unmatched station IDs are listed in the output
- Processing continues even if some files fail

## Performance

- Processes hundreds of files efficiently
- Memory-efficient concatenation of large datasets
- Progress reporting for long-running operations

## Requirements

- Python 3.6+
- pandas
- Standard library modules (os, glob, argparse, pathlib)

## Installation

No special installation required. Just ensure you have pandas installed:
```bash
pip install pandas
```

## Troubleshooting

### Common Issues

1. **"No Bundesland found for station X"**
   - Station ID not found in regions.csv
   - Check if station ID format matches (numeric without leading zeros)

2. **"Error reading file"**
   - CSV file is corrupted or has invalid format
   - Check file encoding and structure

3. **"No data files found"**
   - Input folder doesn't contain CSV files
   - Check folder path and file extensions

### Getting Help

Run the script with `--help` to see all available options:
```bash
python process_weather_by_bundesland.py --help
```
