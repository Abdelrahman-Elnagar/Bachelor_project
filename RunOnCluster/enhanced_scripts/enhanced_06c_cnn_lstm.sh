#!/bin/bash

#SBATCH --job-name=cnn_lstm
#SBATCH --output=cnn_lstm.out
#SBATCH --error=cnn_lstm.err
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=300G
#SBATCH --time=00:30:00

source /opt/bwhpc/common/devel/miniforge/24.11.0-py3.12/etc/profile.d/conda.sh

conda activate bachelor_analysis

cd /home/hu/hu_hu/hu_elnaab01/projects/my_project/Bachelor_project/RunOnCluster/enhanced_scripts

which python
python --version

python enhanced_06c_cnn_lstm.py


