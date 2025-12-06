#!/bin/bash

#SBATCH --job-name=energy_models
#SBATCH --output=train_models.out
#SBATCH --error=train_models.err
#SBATCH --partition=gpu_a100_il     # Changed from dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=300G
#SBATCH --time=24:00:00             # This is now valid (Max is 2 days)

source /opt/bwhpc/common/devel/miniforge/24.11.0-py3.12/etc/profile.d/conda.sh

conda activate bachelor_analysis

cd /home/hu/hu_hu/hu_elnaab01/projects/my_project/Bachelor_project/RunOnCluster/enhanced_scripts

which python
python --version

python train_models.py