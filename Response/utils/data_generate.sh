#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --account=debug
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --array=1
#SBATCH --output=output_log/output_log_%A_%a.out
#SBATCH --error=error_log/error_log_%A_%a.txt

# Create the output_log directory if it doesn't exist
mkdir -p output_log
mkdir -p error_log

# Load the required Python environment
module load conda
conda activate NABC

# Change to the directory where the job was submitted from
SLURM_SUBMIT_DIR=/home/hyun18/NCoin-JDP/Response
cd $SLURM_SUBMIT_DIR


# Calculate seed and prior
seed=$((SLURM_ARRAY_TASK_ID % 100 + 1))
prior_index=$((SLURM_ARRAY_TASK_ID / 100))
priors_list=("P1_0" "P1_1" "P1_2" "P1_3" "P1_4")
prior=${priors_list[$prior_index]}

# Run the Python script with the specified N_EPOCHS value

echo "Running with seed=$seed and prior=$prior"

python utils/calibrate_data_generate.py \
  --seed $seed \
  --priors "$prior"
