import csv
import os
import glob

def remove_eor_column_from_csv(input_file, output_file):
    """
    Remove the 'eor' column from a CSV file.
    """
    with open(input_file, 'r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        
        # Get fieldnames and remove 'eor' if it exists
        fieldnames = [col for col in reader.fieldnames if col != 'eor']
        
        # Read all rows and remove 'eor' column
        rows = []
        for row in reader:
            if 'eor' in row:
                del row['eor']
            rows.append(row)
    
    # Write the modified data back to file
    with open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)
    
    return len(rows)

def remove_eor_from_all_csv_files(directory="."):
    """
    Remove 'eor' column from all CSV files.
    
    NOTE: As of 2025-10-23, weather data files are in the Temp/ directory
    and are named by station ID (e.g., 3987.csv). This script can process
    files in any specified directory.
    """
    
    # Find all CSV files in the specified directory
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    
    print(f"Found {len(csv_files)} CSV files to process in {directory}")
    
    processed_count = 0
    total_records = 0
    
    for csv_file in csv_files:
        try:
            records = remove_eor_column_from_csv(csv_file, csv_file)  # Overwrite the same file
            processed_count += 1
            total_records += records
            print(f"✓ Processed {csv_file} ({records:,} records)")
        except Exception as e:
            print(f"✗ Error processing {csv_file}: {e}")
    
    print(f"\n📊 Processing Summary:")
    print(f"   Files processed: {processed_count}")
    print(f"   Total records: {total_records:,}")
    print(f"   'eor' column removed from all files")

if __name__ == "__main__":
    import sys
    directory = sys.argv[1] if len(sys.argv) > 1 else "."
    remove_eor_from_all_csv_files(directory)
