#!/bin/bash

# Create and activate conda environment
conda env create -f environment.yml
conda activate bachelor_analysis

# Run all scripts
python run_all.py

