#!/bin/bash

# Submit all jobs in sequence with dependencies
# Each job waits for the previous one to complete

echo "Submitting all SLURM jobs in sequence..."

# Submit first job
JOB1=$(sbatch --parsable 01_data_forge.sh)
echo "Submitted job 1 (data_forge): $JOB1"

# Submit subsequent jobs with dependencies
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 02_run_stat.sh)
echo "Submitted job 2 (run_stat): $JOB2"

JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 03_run_ml.sh)
echo "Submitted job 3 (run_ml): $JOB3"

JOB4=$(sbatch --parsable --dependency=afterok:$JOB3 04_run_dl.sh)
echo "Submitted job 4 (run_dl): $JOB4"

JOB5=$(sbatch --parsable --dependency=afterok:$JOB4 05_run_literature.sh)
echo "Submitted job 5 (run_literature): $JOB5"

JOB6=$(sbatch --parsable --dependency=afterok:$JOB5 06_run_legacy.sh)
echo "Submitted job 6 (run_legacy): $JOB6"

echo ""
echo "All jobs submitted successfully!"
echo "Monitor with: squeue -u $USER"

