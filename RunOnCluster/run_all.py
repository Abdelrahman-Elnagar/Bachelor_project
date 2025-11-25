"""
run_all.py
----------
Simple sequential runner for all analysis scripts.
Installs dependencies before each file and runs them in order.
"""

import subprocess
import sys

def pip_install(packages):
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)

def run_script(script_name):
    subprocess.check_call([sys.executable, script_name])

if __name__ == "__main__":
    print("="*60)
    print("STARTING SEQUENTIAL EXECUTION")
    print("="*60)
    
    # 01 - Data Forge
    print("\n[1/6] Running Data Forge...")
    pip_install(["pandas", "numpy", "scikit-learn"])
    run_script("01_data_forge.py")
    
    # 02 - Statistical Models
    print("\n[2/6] Running Statistical Models...")
    pip_install(["pandas", "numpy", "hyperopt", "scikit-learn", "prophet", "tbats", "statsmodels"])
    run_script("02_run_stat.py")
    
    # 03 - Machine Learning Models
    print("\n[3/6] Running ML Models...")
    pip_install(["pandas", "numpy", "xgboost", "lightgbm", "catboost", "hyperopt", "scikit-learn"])
    run_script("03_run_ml.py")
    
    # 04 - Deep Learning Models
    print("\n[4/6] Running DL Models...")
    pip_install(["pandas", "numpy", "torch", "hyperopt", "scikit-learn", "neuralforecast"])
    run_script("04_run_dl.py")
    
    # 05 - Literature Models
    print("\n[5/6] Running Literature Models...")
    pip_install(["pandas", "numpy", "torch", "hyperopt", "scikit-learn"])
    run_script("05_run_literature.py")
    
    # 06 - Legacy Models
    print("\n[6/6] Running Legacy Models...")
    pip_install(["pandas", "numpy", "torch", "hyperopt", "scikit-learn", "neuralforecast"])
    run_script("06_run_legacy.py")
    
    print("\n" + "="*60)
    print("ALL SCRIPTS COMPLETED SUCCESSFULLY")
    print("="*60)

