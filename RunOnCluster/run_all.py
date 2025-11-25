"""
run_all.py
----------
Simple sequential runner for all analysis scripts.
Runs all scripts in sequence (assumes environment is already set up).
"""

import subprocess
import sys

def run_script(script_name):
    subprocess.check_call([sys.executable, script_name])

if __name__ == "__main__":
    print("="*60)
    print("STARTING SEQUENTIAL EXECUTION")
    print("="*60)
    
    print("\n[1/6] Running Data Forge...")
    run_script("01_data_forge.py")
    
    print("\n[2/6] Running Statistical Models...")
    run_script("02_run_stat.py")
    
    print("\n[3/6] Running ML Models...")
    run_script("03_run_ml.py")
    
    print("\n[4/6] Running DL Models...")
    run_script("04_run_dl.py")
    
    print("\n[5/6] Running Literature Models...")
    run_script("05_run_literature.py")
    
    print("\n[6/6] Running Legacy Models...")
    run_script("06_run_legacy.py")
    
    print("\n" + "="*60)
    print("ALL SCRIPTS COMPLETED SUCCESSFULLY")
    print("="*60)

