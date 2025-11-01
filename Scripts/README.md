# Weather Data Batch Processing

This directory contains scripts for processing weather data from multiple folders.

## Scripts

### 1. `process_weather_data.py`
Main processing script that handles individual folders. Performs:
- Downloads files from HTML href links
- Unzips files
- Converts txt to CSV
- Formats date columns
- Renames files with station IDs
- Filters to 2024 data only
- Removes empty files

### 2. `batch_process_weather_folders.py`
Batch processing script that runs the main script on multiple folders sequentially.

## Usage

### Process a single folder:
```bash
python Scripts/process_weather_data.py --folder visibility
```

### Process multiple folders (batch):
```bash
python Scripts/batch_process_weather_folders.py
```

### Custom batch processing:
```bash
python Scripts/batch_process_weather_folders.py --folders visibility wind --base-dir /path/to/data
```

## Dependencies

Install required packages:
```bash
pip install -r Scripts/requirements.txt
```

## Folders Processed

The batch script processes these folders by default:
- `visibility` - Visibility data
- `weather_phenomena` - Weather phenomena data  
- `wind` - Wind data
- `wind_synop` - Wind synoptic data

## Logs

- Individual processing: `weather_data_processing.log`
- Batch processing: `batch_weather_processing.log`
