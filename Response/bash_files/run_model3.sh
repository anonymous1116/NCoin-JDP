#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --account=standby
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --array=10-49            # Create a job array with indices from 1 to 10
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

# Define the starting point for seed 
seed_START=1

# Get the current N_EPOCHS value based on the job array index
seeds=$((seed_START + SLURM_ARRAY_TASK_ID - 1))

TASK="PBJD_summary"  # two_moons, MoG, Lapl, GL_U, slcp, gaussian_mixture, gaussian_linear_uniform, my_five_twomoons, g_and_k
N_EPOCHS=200
layer_len=512
num_training=2000000
tol=.1

# Run the Python script with the specified N_EPOCHS value
# Calculate x0_ind and seed (x0 outer, seed inner)
x0_ind=$((SLURM_ARRAY_TASK_ID / 10))
seed=$((SLURM_ARRAY_TASK_ID % 10 + 1))

echo "Running with seed=$seed, x0_ind=$x0_ind, task=$TASK"

python training_SA6_local.py \
  --experiment "SA6_local" \
  --seed $seed \
  --task $TASK \
  --layer_len $layer_len \
  --num_training $num_training \
  --N_EPOCHS $N_EPOCHS \
  --tol $tol \
  --x0_ind $x0_ind

echo "## Run completed with seed=$seeds, task = $TASK, N_EPCOHS = $N_EPOCHS, layer_len: $layer_len, num_training: $num_training"


