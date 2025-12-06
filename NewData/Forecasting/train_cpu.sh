#!/bin/bash

#SBATCH --job-name=energy_models_cpu
#SBATCH --output=train_models_cpu.out
#SBATCH --error=train_models_cpu.err
#SBATCH --partition=dev_gpu_h100  
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=300G
#SBATCH --time=00:30:00          # 2 days, 23 hours (Max limit is 3-00:00:00)

source /opt/bwhpc/common/devel/miniforge/24.11.0-py3.12/etc/profile.d/conda.sh

conda activate bachelor_analysis

cd /home/hu/hu_hu/hu_elnaab01/projects/my_project/Bachelor_project/RunOnCluster/enhanced_scripts

which python
python --version

python train_models.py