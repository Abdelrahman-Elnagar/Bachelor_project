@echo off

REM Create and activate conda environment
call conda env create -f environment.yml
call conda activate bachelor_analysis

REM Run all scripts
python run_all.py

