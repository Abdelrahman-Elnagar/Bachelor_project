import csv
import os
import glob
from datetime import datetime

def convert_mess_datum_format(input_file, output_file):
    """
    Convert MESS_DATUM from 10-digit format (YYYYMMDDHH) to datetime format.
    Handles files that are already converted and those that need conversion.
    """
    with open(input_file, 'r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        
        # Read all rows
        rows = []
        for row in reader:
            mess_datum = row['MESS_DATUM']
            
            # Check if already in datetime format (contains dashes and colons)
            if '-' in mess_datum and ':' in mess_datum:
                # Already converted, keep as is
                rows.append(row)
            else:
                # Convert from 10-digit format to datetime
                try:
                    # Parse YYYYMMDDHH format
                    if len(mess_datum) == 10 and mess_datum.isdigit():
                        year = int(mess_datum[:4])
                        month = int(mess_datum[4:6])
                        day = int(mess_datum[6:8])
                        hour = int(mess_datum[8:10])
                        
                        # Create datetime object
                        dt = datetime(year, month, day, hour)
                        # Format as YYYY-MM-DD HH:MM:SS
                        row['MESS_DATUM'] = dt.strftime('%Y-%m-%d %H:%M:%S')
                        rows.append(row)
                    else:
                        # Keep original if not in expected format
                        rows.append(row)
                except (ValueError, IndexError):
                    # Keep original if conversion fails
                    rows.append(row)
    
    # Write the converted data back to file
    with open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)
    
    return len(rows)

def convert_all_datetime_formats(directory="Temp"):
    """Convert MESS_DATUM format in all weather CSV files."""
    
    # Find all weather data CSV files in the Temp directory
    # Files are now named by station ID (e.g., 3987.csv, 5100.csv)
    # Also check for old format files that may not have been renamed
    weather_files = glob.glob(os.path.join(directory, "*.csv"))
    weather_files.extend(glob.glob("produkt_tu_stunde_*.csv"))  # Include any non-renamed files
    
    # Remove duplicates
    weather_files = list(set(weather_files))
    
    print(f"Found {len(weather_files)} weather data files to process")
    
    processed_count = 0
    total_records = 0
    already_converted = 0
    newly_converted = 0
    
    for weather_file in weather_files:
        try:
            # Check if file needs conversion by reading first few lines
            needs_conversion = False
            with open(weather_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    if i >= 3:  # Check first few data rows
                        break
                    mess_datum = row['MESS_DATUM']
                    if mess_datum.isdigit() and len(mess_datum) == 10:
                        needs_conversion = True
                        break
            
            if needs_conversion:
                records = convert_mess_datum_format(weather_file, weather_file)
                newly_converted += 1
                print(f"✓ Converted {weather_file} ({records:,} records)")
            else:
                # Count records in already converted file
                with open(weather_file, 'r', encoding='utf-8') as f:
                    records = sum(1 for line in f) - 1  # Subtract header
                already_converted += 1
                print(f"→ Already converted {weather_file} ({records:,} records)")
            
            processed_count += 1
            total_records += records
            
        except Exception as e:
            print(f"✗ Error processing {weather_file}: {e}")
    
    print(f"\n📊 Conversion Summary:")
    print(f"   Files processed: {processed_count}")
    print(f"   Already in datetime format: {already_converted}")
    print(f"   Newly converted: {newly_converted}")
    print(f"   Total records: {total_records:,}")
    print(f"   All MESS_DATUM columns now in YYYY-MM-DD HH:MM:SS format")

if __name__ == "__main__":
    # Process files in the Temp directory (renamed files)
    # Can also specify a different directory if needed
    import sys
    directory = sys.argv[1] if len(sys.argv) > 1 else "Temp"
    convert_all_datetime_formats(directory)
