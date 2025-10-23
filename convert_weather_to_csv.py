import csv
import os
import glob

def convert_weather_file_to_csv(input_file, output_file):
    """
    Convert a German weather data file to CSV format.
    
    Input format:
    STATIONS_ID;MESS_DATUM;QN_9;TT_TU;RF_TU;eor
    3987;1893010101;5;-12.3;84.0;eor
    
    Output format: CSV with proper column headers
    """
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
    
    # Parse header
    header_line = lines[0].strip()
    headers = [col.strip() for col in header_line.split(';')]
    
    # Prepare data
    data = []
    for line in lines[1:]:
        if line.strip():  # Skip empty lines
            # Split by semicolon and clean up
            parts = [part.strip() for part in line.split(';')]
            if len(parts) >= 5:  # Ensure we have enough columns
                data.append(parts)
    
    # Write to CSV
    with open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        writer = csv.writer(outfile, quoting=csv.QUOTE_ALL)
        writer.writerow(headers)
        writer.writerows(data)
    
    return len(data)

def convert_all_weather_files(directory="."):
    """
    Convert all produkt_tu_stunde files to CSV format.
    
    NOTE: As of 2025-10-23, all weather files have been converted to CSV
    and renamed by station ID (e.g., 3987.csv). This script is kept for
    reference or in case new raw .txt files need to be processed.
    """
    
    # Find all produkt files
    produkt_files = glob.glob(os.path.join(directory, "produkt_tu_stunde_*.txt"))
    
    print(f"Found {len(produkt_files)} weather data files to convert")
    
    converted_count = 0
    total_records = 0
    
    for produkt_file in produkt_files:
        # Create output filename
        csv_file = produkt_file.replace('.txt', '.csv')
        
        try:
            records = convert_weather_file_to_csv(produkt_file, csv_file)
            converted_count += 1
            total_records += records
            print(f"✓ Converted {produkt_file} -> {csv_file} ({records:,} records)")
        except Exception as e:
            print(f"✗ Error converting {produkt_file}: {e}")
    
    print(f"\n📊 Conversion Summary:")
    print(f"   Files converted: {converted_count}")
    print(f"   Total records: {total_records:,}")
    print(f"   CSV files created: {converted_count}")

if __name__ == "__main__":
    import sys
    directory = sys.argv[1] if len(sys.argv) > 1 else "."
    convert_all_weather_files(directory)
