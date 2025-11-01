# Extreme Wind Data Processing Script

## Overview
This script processes extreme wind data files sequentially, performing the following operations in order:

1. **Unzip all files** - Extracts all .zip files in the specified folder
2. **Delete zipped files** - Removes all .zip files after extraction
3. **Delete HTML files** - Removes all .html files
4. **Delete meta*.txt files** - Removes all text files starting with "meta"
5. **Convert txt to CSV** - Converts remaining .txt files to .csv format
6. **Format MESS_DATUM** - Converts MESS_DATUM column to proper date format
7. **Rename files** - Renames all CSV files using their STATIONS_ID value
8. **Remove 'eor' column** - Removes the 'eor' column if it exists

## Usage

### Basic Usage (default folder: extreme_wind)
```bash
python Scripts/process_extreme_wind_data.py
```

### Specify Custom Folder
```bash
python Scripts/process_extreme_wind_data.py --folder "path/to/your/folder"
```

### Help
```bash
python Scripts/process_extreme_wind_data.py --help
```

## Features

- **Sequential Processing**: Each step waits for the previous one to complete
- **Configurable Folder Path**: Use `--folder` or `-f` parameter to specify target folder
- **Comprehensive Logging**: Creates detailed logs in `extreme_wind_processing.log`
- **Error Handling**: Continues processing even if individual files fail
- **Date Format Conversion**: Converts MESS_DATUM from YYYYMMDDHH to proper datetime format
- **Smart File Naming**: Renames files using STATIONS_ID for easy identification
- **Data Cleanup**: Removes unnecessary 'eor' column from all files

## Requirements

- Python 3.6+
- pandas
- Standard library modules (os, zipfile, glob, pathlib, argparse, logging, datetime)

## Example Output

The script will process files like:
- `stundenwerte_FX_19852_20231011_20241231_hist.zip` → extracts to multiple files
- Deletes metadata files (HTML and meta*.txt)
- Converts `produkt_fx_stunde_20231011_20241231_19852.txt` → `produkt_fx_stunde_20231011_20241231_19852.csv`
- Formats MESS_DATUM column from `2023101119` to `2023-10-11 19:00:00`
- Renames file to `19852.csv` (using STATIONS_ID)
- Removes 'eor' column from the final CSV

## Logging

The script creates detailed logs showing:
- Number of files processed at each step
- Success/failure status for each file
- Error messages for any failures
- Overall completion status
