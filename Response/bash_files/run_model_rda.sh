#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --account=standby
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --array=0-49%25
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
seed=$((SLURM_ARRAY_TASK_ID % 10 + 1))
prior_index=$((SLURM_ARRAY_TASK_ID / 10))
priors_list=("P3_0" "P3_1" "P3_2" "P3_3" "P3_4")
prior=${priors_list[$prior_index]}

N_EPOCHS=200
layer_len=512
num_training=2000000
tol=.1
TASK="PBJD_summary"
# Run the Python script with the specified N_EPOCHS value

echo "Running with seed=$seed and prior=$prior"

python training_SA6_priors.py \
  --experiment "SA6_priors" \
  --seed $seed \
  --task $TASK \
  --layer_len $layer_len \
  --num_training $num_training \
  --N_EPOCHS $N_EPOCHS \
  --tol $tol \
  --priors "$prior"
