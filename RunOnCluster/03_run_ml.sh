#!/bin/bash

#SBATCH --job-name=run_ml
#SBATCH --output=run_ml.out
#SBATCH --error=run_ml.err
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=300G
#SBATCH --time=72:00:00

source ~/miniconda3/etc/profile.d/conda.sh

conda activate bachelor_analysis

cd /home/hu/hu_hu/hu_elnaab01/projects/my_project/Bachelor_project/RunOnCluster

which python
python --version

python 03_run_ml.py

