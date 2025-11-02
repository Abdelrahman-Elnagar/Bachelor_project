# Weather Data Preprocessing and Visualization

This directory contains scripts for comprehensive preprocessing and visualization of German weather data.

## Overview

The preprocessing and visualization pipeline processes weather data files in-place (to save disk space) and generates comprehensive visualizations including:
- Monthly plots for each attribute (12 plots per attribute)
- Full-year plots showing all 8760 hours
- Distribution comparison plots
- Statistical summary tables

## Scripts

### 1. `preprocess_weather_data.py`

Main preprocessing script that handles:
- **Missing Values**: Replaces -999 and -999.0 with NaN
- **Outlier Detection**: Flags outliers using domain-specific thresholds and statistical methods (Z-score, IQR) - **does not remove rows**, adds `outlier_flag` column
- **Data Type Validation**: Ensures correct data types (numeric, datetime)
- **Timestamp Validation**: Verifies MESS_DATUM is valid datetime and within 2024
- **Duplicate Detection**: Identifies and logs duplicate records
- **Data Range Validation**: Checks values are within reasonable physical bounds
- **Completeness Checks**: Calculates data completeness percentage per file/attribute
- **In-Place Processing**: Modifies CSV files directly (no separate output directory)

**Usage:**
```bash
python Scripts/preprocess_weather_data.py
```

The script processes all attribute folders in `Data/Bundesland_aggregation/` and `Data/Germany_aggregation/`.

**Output:**
- Preprocessed CSV files (modified in-place)
- Summary reports in `Scripts/preprocessing_reports/`:
  - `preprocessing_summary_YYYY-MM-DD_HH-MM-SS.json` - Detailed JSON report
  - `preprocessing_summary_YYYY-MM-DD_HH-MM-SS.csv` - CSV summary

### 2. `visualize_weather_comprehensive.py`

Comprehensive visualization script that generates:

**Monthly Visualizations** (12 plots per attribute):
- Line charts showing hourly data for each month (Jan-Dec)
- Each Bundesland as a separate line + Germany aggregation line (dashed, thicker)
- X-axis: Hour of month (0-744 hours), Y-axis: Attribute value
- Saved as: `visualizations/{attribute}/monthly/{attribute}_month_{01-12}_2024.png`

**Full Year Visualization** (1 plot per attribute):
- Line chart showing all 8760 hours of 2024
- Each Bundesland as a line + Germany aggregation line
- X-axis: Hour of year (0-8760), Y-axis: Attribute value
- Saved as: `visualizations/{attribute}/{attribute}_full_year_2024.png`

**Distribution Comparisons** (12 plots per attribute):
- Monthly violin plots comparing each Bundesland vs Germany aggregation
- Shows distributions side-by-side for each month
- Saved as: `visualizations/{attribute}/distributions/{attribute}_distribution_month_{01-12}_2024.png`

**Statistical Summary Tables**:
- Monthly statistics (mean, std, min, max, count) for each Bundesland and Germany
- Saved as: `visualizations/{attribute}/summaries/{attribute}_monthly_stats_2024.csv`

**Usage:**
```bash
python Scripts/visualize_weather_comprehensive.py
```

**Output Structure:**
```
visualizations/
  {attribute}/             # One folder per attribute
    monthly/              # 12 monthly plots
    distributions/         # Monthly distribution comparisons
    summaries/            # Statistical summary CSVs
    {attribute}_full_year_2024.png
```

### 3. `preprocessing_config.yaml`

Configuration file containing:
- Missing value markers (-999, -999.0, etc.)
- Domain-specific thresholds for outlier detection (per attribute)
- Z-score and IQR thresholds for outlier flagging
- Visualization parameters (colors, figure sizes, line styles)
- Attribute-specific settings and column mappings

## Configuration

### Outlier Detection

Outliers are flagged (not removed) using three methods:

1. **Domain-Specific Thresholds**: Values outside physical bounds (e.g., temperature < -50°C or > 50°C)
2. **Z-Score Method**: Values > 3 standard deviations from mean
3. **IQR Method**: Values outside 1.5 × IQR from Q1/Q3

All flagged outliers are marked in the `outlier_flag` column.

### Attribute Thresholds

Default thresholds (can be modified in `preprocessing_config.yaml`):

- **Temperature**: -50°C to 50°C
- **Wind Speed**: 0 to 200 km/h
- **Precipitation**: 0 to 500 mm
- **Pressure**: 800 to 1100 hPa
- **Cloudiness**: 0 to 8 oktas
- **Dew Point**: -50°C to 50°C
- **Visibility**: 0 to 100 km
- **Humidity**: 0 to 100%
- **Soil Temperature**: -30°C to 60°C
- **Sunshine**: 0 to 24 hours

## Requirements

Install dependencies:
```bash
pip install -r Scripts/requirements.txt
```

Required packages:
- pandas >= 1.3.0
- numpy >= 1.20.0
- matplotlib >= 3.3.0
- seaborn >= 0.11.0
- scipy >= 1.6.0
- pyyaml >= 5.4.0

## Workflow

### Step 1: Preprocess Data

Run preprocessing to clean data and flag outliers:

```bash
python Scripts/preprocess_weather_data.py
```

This will:
1. Process all CSV files in Bundesland and Germany aggregation folders
2. Replace missing values (-999) with NaN
3. Flag outliers (add `outlier_flag` column)
4. Validate data types and timestamps
5. Generate preprocessing reports

### Step 2: Generate Visualizations

After preprocessing, generate all visualizations:

```bash
python Scripts/visualize_weather_comprehensive.py
```

This will create:
- 12 monthly plots per attribute
- 1 full-year plot per attribute
- 12 distribution plots per attribute
- 1 statistical summary CSV per attribute

## Attributes Supported

The pipeline supports 12 weather attributes:

1. **temperature** - Temperature (mean, min, max)
2. **cloudiness** - Cloud cover (oktas)
3. **wind** - Wind speed and direction
4. **wind_synop** - Synoptic wind data
5. **precipitation** - Rainfall (mm)
6. **pressure** - Atmospheric pressure (hPa)
7. **dew_point** - Dew point temperature
8. **moisture** - Relative humidity
9. **extreme_wind** - Peak wind measurements
10. **soil_temperature** - Ground temperature
11. **sun** - Sunshine duration
12. **visibility** - Horizontal visibility

## Preprocessing Reports

Preprocessing reports are saved in `Scripts/preprocessing_reports/` and include:

- **Files processed**: Total number of files processed
- **Total rows**: Total number of data rows
- **Missing values replaced**: Count of -999 values replaced with NaN
- **Outliers flagged**: Count of rows flagged as outliers
- **Duplicates found**: Count of duplicate records
- **Validation errors**: List of any errors encountered

Each file processed also includes:
- Rows processed
- Missing values replaced
- Outliers flagged
- Duplicates found
- Data completeness percentage
- Any validation errors

## Visualization Output

### Monthly Plots

Each monthly plot shows:
- Hourly data points for all 16 Bundesländer (colored lines)
- Germany aggregation (black dashed line, thicker)
- Month name and attribute name in title
- Legend with Bundesland codes

### Full-Year Plots

Full-year plots show:
- All 8760 hours of 2024
- All Bundesländer lines
- Germany aggregation line
- Month boundary markers (vertical dashed lines)

### Distribution Plots

Distribution plots show:
- Violin plots for each Bundesland
- Germany aggregation distribution
- Mean and median markers
- Monthly comparison across regions

### Statistical Summaries

Summary CSV files contain:
- Month
- Region (Bundesland code or "Germany")
- Region Type (Bundesland or Germany)
- Mean, Std, Min, Max, Count

## Troubleshooting

### Missing Data

If visualizations show missing data:
1. Check preprocessing reports for completeness statistics
2. Verify data files exist in expected folders
3. Check that column names match those in `preprocessing_config.yaml`

### Outlier Flags

To see which rows are flagged as outliers:
```python
import pandas as pd
df = pd.read_csv("path/to/file.csv")
outliers = df[df['outlier_flag'] == True]
```

### Configuration Errors

If attributes fail to visualize:
1. Check `preprocessing_config.yaml` has correct folder names
2. Verify column mappings match actual data columns
3. Ensure value columns exist in data files

## Notes

- **In-Place Processing**: CSV files are modified directly to save disk space. Original data is overwritten.
- **Outlier Flagging**: Outliers are flagged but not removed. Use `outlier_flag` column to filter if needed.
- **Missing Values**: -999 values are replaced with NaN. Handle NaN values appropriately in analysis.
- **Large Files**: Some attribute files are very large. Processing may take time.

