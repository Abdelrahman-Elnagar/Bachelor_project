import os
import csv
import glob
from collections import defaultdict

def verify_weather_data_integrity(directory="Temp"):
    """
    Verify the integrity of weather data files and provide a summary.
    
    Checks:
    - Number of CSV files in Temp directory
    - Matches with regions.csv station IDs
    - Identifies corrupted files (old naming pattern)
    - Validates CSV structure
    - Reports data coverage
    """
    print("=" * 70)
    print("WEATHER DATA INTEGRITY VERIFICATION")
    print("=" * 70)
    
    # 1. Check Temp directory
    print(f"\n1. Checking {directory}/ directory...")
    if not os.path.exists(directory):
        print(f"   ✗ ERROR: {directory}/ directory not found!")
        return
    
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    print(f"   ✓ Found {len(csv_files)} CSV files")
    
    # 2. Identify renamed vs original format files
    renamed_files = [f for f in csv_files if not os.path.basename(f).startswith("produkt_")]
    original_files = [f for f in csv_files if os.path.basename(f).startswith("produkt_")]
    
    print(f"   ✓ Renamed files (station ID format): {len(renamed_files)}")
    if original_files:
        print(f"   ⚠ Files with original naming (possibly corrupted): {len(original_files)}")
        for f in original_files:
            print(f"      - {os.path.basename(f)}")
    
    # 3. Check regions.csv
    print(f"\n2. Checking regions.csv...")
    if not os.path.exists("regions.csv"):
        print(f"   ✗ ERROR: regions.csv not found!")
        return
    
    with open("regions.csv", 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        regions = list(reader)
    
    print(f"   ✓ Loaded {len(regions)} stations from regions.csv")
    
    # 4. Match station IDs
    print(f"\n3. Matching station IDs with weather files...")
    station_ids_in_csv = set()
    for f in renamed_files:
        basename = os.path.basename(f)
        station_id = basename.replace('.csv', '')
        if station_id.isdigit():
            station_ids_in_csv.add(station_id)
    
    region_station_ids = {str(int(r['Stations_id'])) for r in regions}
    
    matched = station_ids_in_csv & region_station_ids
    in_files_not_regions = station_ids_in_csv - region_station_ids
    in_regions_not_files = region_station_ids - station_ids_in_csv
    
    print(f"   ✓ Matched stations: {len(matched)}")
    print(f"   ⚠ Stations in files but not in regions.csv: {len(in_files_not_regions)}")
    if in_files_not_regions:
        print(f"      IDs: {sorted(in_files_not_regions)[:10]}{'...' if len(in_files_not_regions) > 10 else ''}")
    
    print(f"   ⚠ Stations in regions.csv but no data file: {len(in_regions_not_files)}")
    if in_regions_not_files:
        print(f"      IDs: {sorted(in_regions_not_files)[:10]}{'...' if len(in_regions_not_files) > 10 else ''}")
    
    # 5. Validate CSV structure (sample check)
    print(f"\n4. Validating CSV structure (sampling 10 files)...")
    sample_files = renamed_files[:10] if len(renamed_files) >= 10 else renamed_files
    valid_count = 0
    
    expected_columns = ['STATIONS_ID', 'MESS_DATUM', 'QN_9', 'TT_TU', 'RF_TU']
    
    for csv_file in sample_files:
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames
                
                # Check if all expected columns exist
                if all(col in headers for col in expected_columns):
                    # Check if there's no 'eor' column
                    if 'eor' not in headers:
                        valid_count += 1
                    else:
                        print(f"   ⚠ File still has 'eor' column: {os.path.basename(csv_file)}")
                else:
                    missing = [col for col in expected_columns if col not in headers]
                    print(f"   ✗ Missing columns in {os.path.basename(csv_file)}: {missing}")
        except Exception as e:
            print(f"   ✗ Error reading {os.path.basename(csv_file)}: {e}")
    
    print(f"   ✓ Valid files in sample: {valid_count}/{len(sample_files)}")
    
    # 6. Summary
    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total weather files: {len(csv_files)}")
    print(f"  - Renamed (clean): {len(renamed_files)}")
    print(f"  - Original format (possibly corrupted): {len(original_files)}")
    print(f"Total stations in regions.csv: {len(regions)}")
    print(f"Matched stations: {len(matched)}")
    print(f"Coverage: {len(matched)/len(regions)*100:.1f}%")
    
    if original_files:
        print(f"\n⚠ ACTION REQUIRED:")
        print(f"   - {len(original_files)} files need attention (may be corrupted)")
        print(f"   - Consider re-downloading or manually fixing these files")
    
    print("=" * 70)

def verify_cloud_data():
    """Verify cloud type data integrity."""
    print("\n" + "=" * 70)
    print("CLOUD TYPE DATA VERIFICATION")
    print("=" * 70)
    
    # Check Cloud_type directory
    print(f"\n1. Checking Cloud_type/ directory...")
    if not os.path.exists("Cloud_type"):
        print(f"   ✗ ERROR: Cloud_type/ directory not found!")
        return
    
    txt_files = glob.glob("Cloud_type/*.txt")
    print(f"   ✓ Found {len(txt_files)} cloud data files")
    
    # Check cloudtypedata.csv
    print(f"\n2. Checking cloudtypedata.csv...")
    if not os.path.exists("cloudtypedata.csv"):
        print(f"   ⚠ WARNING: cloudtypedata.csv not found!")
        return
    
    with open("cloudtypedata.csv", 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        cloud_inventory = list(reader)
    
    print(f"   ✓ Loaded {len(cloud_inventory)} entries from cloudtypedata.csv")
    
    # Summary
    print(f"\n" + "=" * 70)
    print("CLOUD DATA SUMMARY")
    print("=" * 70)
    print(f"Expected files (from CSV): {len(cloud_inventory)}")
    print(f"Files present: {len(txt_files)}")
    print(f"Missing files: {len(cloud_inventory) - len(txt_files)}")
    print(f"Completion: {len(txt_files)/len(cloud_inventory)*100:.1f}%")
    print("=" * 70)

if __name__ == "__main__":
    import sys
    
    # Allow specifying directory as command line argument
    weather_dir = sys.argv[1] if len(sys.argv) > 1 else "Temp"
    
    verify_weather_data_integrity(weather_dir)
    verify_cloud_data()
    
    print("\nVerification complete!")

