# Weather Data Preprocessing - Complete Guide

This document provides instructions for using all preprocessing scripts for the German weather station dataset.

## Dataset Overview

- **Weather Stations**: 636 stations across Germany
- **Weather Data Files**: 633 CSV files (99.5% complete)
- **Cloud Type Data**: 320 files (53.7% complete - 276 files missing)
- **Date Range**: 1893-2024 (131 years)
- **Total Records**: 151+ million hourly measurements
- **Parameters**: Temperature (TT_TU), Humidity (RF_TU), Quality flags (QN_9)

## Directory Structure

```
Bachelor abroad/
├── Temp/                       # Weather data files (named by station ID)
│   ├── 3987.csv
│   ├── 5100.csv
│   └── ... (633 files)
├── Cloud_type/                 # Cloud observation data files
│   └── produkt_cs_stunde_*.txt (320 files)
├── regions.csv                 # Station metadata
├── stations_data.csv           # Data file inventory
├── cloudtypedata.csv           # Cloud data inventory
├── missing_cloud_files.csv     # List of missing cloud files
└── [preprocessing scripts]
```

## Preprocessing Scripts

### 1. convert_weather_to_csv.py
**Purpose**: Convert raw weather data from semicolon-delimited format to CSV

**Usage**:
```bash
# Process files in current directory
python convert_weather_to_csv.py

# Process files in specific directory
python convert_weather_to_csv.py /path/to/directory
```

**Note**: All weather files have already been converted. Keep this script for processing new raw .txt files.

---

### 2. remove_eor_column.py
**Purpose**: Remove the unnecessary 'eor' column from CSV files

**Usage**:
```bash
# Process files in current directory
python remove_eor_column.py

# Process files in Temp directory
python remove_eor_column.py Temp
```

**Note**: All current weather files have already had the 'eor' column removed.

---

### 3. convert_datetime_format.py
**Purpose**: Convert MESS_DATUM from YYYYMMDDHH format to YYYY-MM-DD HH:MM:SS format

**Input Format**: `1893010101` (10-digit format)
**Output Format**: `1893-01-01 01:00:00` (datetime format)

**Usage**:
```bash
# Process files in Temp directory (default)
python convert_datetime_format.py

# Process files in specific directory
python convert_datetime_format.py /path/to/directory
```

**Features**:
- Automatically detects if files are already converted
- Skips files that don't need conversion
- Provides detailed summary of conversions

---

### 4. data_verification.py
**Purpose**: Verify data integrity and provide comprehensive summary

**Usage**:
```bash
# Verify weather data in Temp directory (default)
python data_verification.py

# Verify weather data in specific directory
python data_verification.py /path/to/directory
```

**Checks Performed**:
- Number of CSV files in Temp directory
- Matches with regions.csv station IDs
- Identifies corrupted files
- Validates CSV structure (headers, columns)
- Reports data coverage percentage
- Verifies cloud type data completeness

**Output Example**:
```
Total weather files: 633
  - Renamed (clean): 633
  - Original format (possibly corrupted): 0
Total stations in regions.csv: 636
Matched stations: 633
Coverage: 99.5%
```

---

## Data Files

### regions.csv
**Description**: Station metadata with geographic information

**Columns**:
- `Stations_id`: Unique station identifier
- `von_datum`: Start date of station operation
- `bis_datum`: End date of station operation
- `Stationshoehe`: Station elevation (meters)
- `geoBreite`: Latitude
- `geoLaenge`: Longitude
- `Stationsname`: Station name
- `Bundesland`: German state

**Records**: 636 stations

---

### Weather Data Files (Temp/*.csv)
**Description**: Hourly temperature and humidity measurements per station

**Naming Convention**: `{station_id}.csv` (e.g., `3987.csv`, `5100.csv`)

**Columns**:
- `STATIONS_ID`: Station identifier (matches regions.csv)
- `MESS_DATUM`: Measurement datetime (YYYY-MM-DD HH:MM:SS)
- `QN_9`: Quality flag
- `TT_TU`: Temperature (°C)
- `RF_TU`: Relative humidity (%)

**File Count**: 633 files

---

### Cloud Type Data Files (Cloud_type/*.txt)
**Description**: Hourly cloud observation data

**Naming Convention**: `produkt_cs_stunde_{start_date}_{end_date}_{station_id}.txt`

**File Count**: 320 files (276 missing)

---

### cloudtypedata.csv
**Description**: Inventory of all cloud type data files

**Records**: 596 entries

---

### missing_cloud_files.csv
**Description**: List of cloud data files that are missing from Cloud_type directory

**Columns**:
- `csv_name`: Original CSV entry
- `expected_name`: Expected filename in Cloud_type folder
- `station_id`: Station identifier
- `date_range`: Date range for the data

**Records**: 276 missing files

---

## Known Issues

### Corrupted Weather Files
Three weather station files could not be processed due to NUL character corruption:

1. **Station 400**: File contains NUL characters
2. **Station 704**: File contains NUL characters  
3. **Station 840**: File contains NUL characters

**Resolution**: These files need to be re-downloaded from the original source or manually fixed.

### Missing Cloud Data
276 cloud observation files are missing from the dataset (46% of total inventory).

**Action Required**: Extract remaining zip files or obtain missing data to complete the dataset.

---

## Complete Preprocessing Workflow

If starting from raw data, follow these steps in order:

1. **Extract weather data zip files** → Raw .txt files
2. **Convert to CSV format**:
   ```bash
   python convert_weather_to_csv.py
   ```
3. **Remove unnecessary columns**:
   ```bash
   python remove_eor_column.py Temp
   ```
4. **Convert datetime format**:
   ```bash
   python convert_datetime_format.py Temp
   ```
5. **Rename files by station ID** (already done)
6. **Verify data integrity**:
   ```bash
   python data_verification.py
   ```

---

## Data Quality

### Weather Data
- ✅ **99.5% Complete** (633 of 636 stations)
- ✅ **CSV Format** with proper headers
- ✅ **Datetime Format** converted
- ✅ **Clean Structure** (no unnecessary columns)
- ✅ **Organized** by station ID
- ⚠️ **3 Corrupted Files** need attention

### Cloud Data
- ⚠️ **53.7% Complete** (320 of 596 files)
- ⚠️ **276 Missing Files** need extraction
- ✅ **Inventory Available** (cloudtypedata.csv)
- ✅ **Missing List Available** (missing_cloud_files.csv)

---

## For Data Analysis

The dataset is ready for:
- Weather pattern analysis
- Climate research
- Time series forecasting
- Machine learning applications
- Geographic weather modeling

**Recommendation**: Complete the cloud type dataset extraction before final analysis for comprehensive weather characterization.

---

## Support Files

- **data_processing_steps.txt**: Detailed log of all preprocessing steps
- **stations_data.csv**: Complete inventory of weather data files
- **README_PREPROCESSING.md**: This file

---

## Last Updated
2025-10-23

## Version
Dataset v1.0 - Weather data complete (99.5%), Cloud data partial (53.7%)

