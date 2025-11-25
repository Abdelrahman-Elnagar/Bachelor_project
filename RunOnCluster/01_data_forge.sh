#!/bin/bash

#SBATCH --job-name=data_forge
#SBATCH --output=data_forge.out
#SBATCH --error=data_forge.err
#SBATCH --partition=gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=300G
#SBATCH --time=3:00:00

source ~/miniconda3/etc/profile.d/conda.sh

conda activate bachelor_analysis

cd /home/hu/hu_hu/hu_elnaab01/projects/my_project/Bachelor_project/RunOnCluster

which python
python --version

python 01_data_forge.py

