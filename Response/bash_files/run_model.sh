#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --account=debug
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --array=1-2           # Create a job array with indices from 1 to 10
#SBATCH --output=output_log/output_log_%A_%a.out
#SBATCH --error=error_log/error_log_%A_%a.txt

# Create the output_log directory if it doesn't exist
mkdir -p output_log

# Load the required Python environment
module use /depot/wangxiao/etc/modules
module load conda-env/sbi_pack-py3.11.7

# Change to the directory where the job was submitted from
SLURM_SUBMIT_DIR=/home/hyun18/NCoin-JDP/Response
cd $SLURM_SUBMIT_DIR

# Define the starting point for seed 
seed_START=1

# Get the current N_EPOCHS value based on the job array index
seeds=$((seed_START + SLURM_ARRAY_TASK_ID - 1))

TASK="OU_summary"  # two_moons, MoG, Lapl, GL_U, slcp, gaussian_mixture, gaussian_linear_uniform, my_five_twomoons, g_and_k
N_EPOCHS=200
layer_len=256
num_training=100000
#num_calibrations=100000000
#num_calibrations=1000000000


# Run the Python script with the specified N_EPOCHS value
echo "Running with seed=$seeds, task = $TASK, N_EPCOHS = $N_EPOCHS, layer_len: $layer_len, num_training: $num_training"
#python training_SA1.py --experiment "SA1" --seed $seeds --task $TASK --layer_len $layer_len --num_training $num_training --N_EPOCHS $N_EPOCHS
python utils.creating_training.py --experiment "SA1" --task $TASK --layer_len $layer_len --num_training $num_training --N_EPOCHS $N_EPOCHS
echo "## Run completed with seed=$seeds, task = $TASK, N_EPCOHS = $N_EPOCHS, layer_len: $layer_len, num_training: $num_training"
