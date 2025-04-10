#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --account=standby
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --array=0-3
#SBATCH --output=output_log/output_log_%A_%a.out
#SBATCH --error=error._log/error_log_%A_%a.txt

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
prior_index=$((SLURM_ARRAY_TASK_ID))
#priors_list=("P1_0" "P1_1" "P1_2" "P1_3" "P1_4" "P2_0" "P2_1" "P2_2" "P2_3" "P2_4" "P3_0" "P3_1" "P3_2" "P3_3" "P3_4" "P4_0" "P4_1" "P4_2" "P4_3" "P4_4")
priors_list=("P1_4" "P2_4" "P3_4" "P4_4")

prior=${priors_list[$prior_index]}

# Run the Python script with the specified N_EPOCHS value

echo "Running with prior=$prior"

python utils/calibrate_summary.py \
  --priors "$prior"

echo "Running finished with prior=$prior"
