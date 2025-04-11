#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --account=standby
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --array=0-9
#SBATCH --output=output_log/output_log_%A_%a.out
#SBATCH --error=error_log/error_log_%A_%a.txt

# Create output directories
mkdir -p output_log
mkdir -p error_log

# Load environment
module load conda
conda activate NABC

# Move to working directory
cd /home/hyun18/NCoin-JDP/Response

# Set fixed prior and get x0_ind from SLURM array index
prior="P1_0"
x0_ind=$SLURM_ARRAY_TASK_ID

echo "Running with prior=$prior and x0_ind=$x0_ind"

python utils/calibrate_summary.py \
  --priors "$prior" \
  --x0_ind "$x0_ind"

echo "Finished with prior=$prior and x0_ind=$x0_ind"
