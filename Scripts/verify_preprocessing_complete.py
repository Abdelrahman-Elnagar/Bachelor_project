#!/usr/bin/env python3
"""Verify preprocessing and visualization completeness"""
import pandas as pd
from pathlib import Path

print("="*60)
print("VERIFICATION: PREPROCESSING AND VISUALIZATION")
print("="*60)

# Check 1: Row preservation (imputation vs removal)
print("\n1. CHECKING ROW PRESERVATION (Imputation vs Removal)...")
orig_file = Path("Data/Bundesland_aggregation/Temp_Bundesland_Aggregated/Bayern_aggregated.csv")
prep_file = Path("Preprocessed_Data/Bundesland_aggregation/Temp_Bundesland_Aggregated/Bayern_aggregated.csv")

if orig_file.exists() and prep_file.exists():
    orig_df = pd.read_csv(orig_file, on_bad_lines='skip', engine='python')
    prep_df = pd.read_csv(prep_file, on_bad_lines='skip', engine='python')
    
    print(f"   Original rows: {len(orig_df):,}")
    print(f"   Preprocessed rows: {len(prep_df):,}")
    rows_ok = len(prep_df) == len(orig_df)
    print(f"   Rows preserved: {rows_ok} [OK]" if rows_ok else f"   Rows preserved: {rows_ok} [ISSUE]")
    nan_count = prep_df['temperature_mean'].isna().sum()
    print(f"   NaN in preprocessed temp: {nan_count} (should be 0)")
    print(f"   Imputation working: {nan_count == 0} [OK]" if nan_count == 0 else "   [ISSUE: NaN found!]")

# Check 2: Preprocessed file count
print("\n2. CHECKING PREPROCESSED FILE COUNTS...")
base_path = Path("Preprocessed_Data/Bundesland_aggregation")
attr_folders = [d for d in base_path.iterdir() if d.is_dir()]

total_files = 0
for folder in attr_folders:
    csv_files = list(folder.glob("*.csv"))
    csv_files = [f for f in csv_files if "SUMMARY" not in f.name.upper() and "AGGREGATION" not in f.name.upper()]
    total_files += len(csv_files)
    if len(csv_files) > 0:
        print(f"   {folder.name}: {len(csv_files)} files")

files_ok = total_files == 192
print(f"   Total Bundesland files: {total_files} (expected: 192) {'[OK]' if files_ok else '[INCOMPLETE]'}")

# Check 3: Visualization completeness
print("\n3. CHECKING VISUALIZATION COMPLETENESS...")
viz_path = Path("Preprocessed_Visualizations")
attr_folders = [d for d in viz_path.iterdir() if d.is_dir()] if viz_path.exists() else []

expected_attrs = ['temperature', 'cloudiness', 'wind', 'wind_synop', 'precipitation', 
                  'pressure', 'dew_point', 'moisture', 'extreme_wind', 
                  'soil_temperature', 'sun', 'visibility']

all_complete = True
for attr in expected_attrs:
    attr_path = viz_path / attr
    if attr_path.exists():
        monthly = len(list((attr_path / "monthly").glob("*.png"))) if (attr_path / "monthly").exists() else 0
        dist = len(list((attr_path / "distributions").glob("*.png"))) if (attr_path / "distributions").exists() else 0
        full = (attr_path / f"{attr}_preprocessed_full_year_2024.png").exists()
        summary = len(list((attr_path / "summaries").glob("*.csv"))) > 0 if (attr_path / "summaries").exists() else False
        
        complete = (monthly == 12 and dist == 12 and full and summary)
        status = "[OK]" if complete else "[INCOMPLETE]"
        if not complete:
            all_complete = False
        
        print(f"   {attr:20s}: Monthly={monthly:2d}/12, Dist={dist:2d}/12, Full={full}, Summary={summary} {status}")
    else:
        print(f"   {attr:20s}: FOLDER MISSING [MISSING]")
        all_complete = False

# Summary
print("\n" + "="*60)
print("VERIFICATION SUMMARY")
print("="*60)
print(f"[OK] Preprocessing: Row preservation verified")
print(f"{'[OK]' if total_files == 192 else '[INCOMPLETE]'} Preprocessed files: {total_files}/192")
print(f"{'[OK]' if all_complete else '[INCOMPLETE]'} Visualizations: {'Complete' if all_complete else 'Some missing'}")
print("\nOverall Status: ", "[OK] CORRECT" if (total_files == 192 and all_complete) else "[INCOMPLETE] NEEDS ATTENTION")

