#!/usr/bin/env python3
"""
Check if all required dependencies are installed
"""

import sys

required_packages = {
    'pandas': 'pandas',
    'numpy': 'numpy',
    'scipy': 'scipy',
    'sklearn': 'scikit-learn',
    'lightgbm': 'lightgbm',
    'xgboost': 'xgboost',
    'catboost': 'catboost',
    'optuna': 'optuna',
    'torch': 'torch',
    'prophet': 'prophet',
    'statsmodels': 'statsmodels',
    'shap': 'shap',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',
    'yaml': 'pyyaml',
    'pytz': 'pytz'
}

print("="*60)
print("CHECKING DEPENDENCIES")
print("="*60)

missing = []
installed = []

for import_name, package_name in required_packages.items():
    try:
        __import__(import_name)
        installed.append(package_name)
        print(f"OK  {package_name:20s} - Installed")
    except ImportError:
        missing.append(package_name)
        print(f"NO  {package_name:20s} - NOT INSTALLED")

print("\n" + "="*60)
print(f"Summary: {len(installed)}/{len(required_packages)} packages installed")
print("="*60)

if missing:
    print("\nMissing packages. Install with:")
    print(f"pip install {' '.join(missing)}")
    print("\nOr install all at once:")
    print("pip install -r requirements.txt")
    sys.exit(1)
else:
    print("\nOK All dependencies are installed!")
    sys.exit(0)

