import os
import numpy as np
import subprocess
import argparse
import sys
# Add the parent directory to the system path to import simulator.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from simulator import get_task_parameters 

def create_training_job_script(experiment, task, num_training, N_EPOCHS, seed, layer_len, calibrate, num_calibrations, iter_calibrations, c2st):
    calibrate_flag = "--calibrate" if calibrate else ""
    c2st_flag = "--c2st" if c2st else ""
    job_script = f"""#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --account=debug
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --output=output_log/{experiment}/{task}/output_log_{seed}%A.log
#SBATCH --error=output_log/{experiment}/{task}/error_log_{seed}%A.txt

# Create the output_log directory if it doesn't exist
mkdir -p output_log/{experiment}/{task}

# Load the required Python environment
module use /depot/wangxiao/etc/modules
module load conda-env/sbi_pack-py3.11.7

# Change to the directory where the job was submitted from
SLURM_SUBMIT_DIR=$(pwd)
cd $SLURM_SUBMIT_DIR

# Run the Python script for the current simulation
echo "Running training for task task: '{task}', 'num_training: {num_training}', N_EPOCHS: {N_EPOCHS} seed: {seed} layer_len={layer_len} calibrate = {calibrate_flag} num_calibrations = {num_calibrations}"
python training_SA.py --experiment {experiment} --task {task} --num_training {num_training} --N_EPOCHS {N_EPOCHS} --seed {seed} --layer_len {layer_len} {calibrate_flag} --num_calibrations {num_calibrations} --iter_calibrations {iter_calibrations} {c2st_flag}
echo "Training completed task: '{task}', 'num_training: {num_training}', N_EPOCHS: {N_EPOCHS} seed: {seed} layer_len={layer_len} calibrate = {calibrate_flag} num_calibrations = {num_calibrations}"
"""
    # Create the directory for SLURM files if it doesn't exist
    output_dir = f"../../depot_hyun/hyun/NCoinJDP/{experiment}/{task}/J_{int(num_training/1000)}K/slurm_files"
    os.makedirs(output_dir, exist_ok=True)

    job_file_path = os.path.join(output_dir, f"{task}_train_{num_training}_{seed}.sh")
    with open(job_file_path, 'w') as f:
        f.write(job_script)
    print(f"Job script created: {job_file_path}")

    # Submit the job immediately
    subprocess.run(['sbatch', job_file_path])
    print(f"Job {job_file_path} submitted.")

def main(args):
    for j in range(1, 3):
    #for j in range(1, 3):    
        create_training_job_script(args.experiment, args.task, args.num_training, args.N_EPOCHS, j, args.layer_len, args.calibrate, args.num_calibrations, args.iter_calibrations, args.c2st)    
        print(f"create training job script task: '{args.task}', 'num_training: {args.num_training}', N_EPOCHS: {args.N_EPOCHS} seed: {args.seed} layer_len={args.layer_len} num_calibrations = {args.num_calibrations}")
def get_args():
    parser = argparse.ArgumentParser(description="Run simulation with customizable parameters.")
    parser.add_argument('--experiment', type=str, default='SA1', 
                        help='experiment type: S1 ...')
    parser.add_argument('--task', type=str, default='OU', 
                        help='Simulation type: OU, CIR ...')
    parser.add_argument("--num_training", type=int, default=500_000,
                        help="Number of simulations for training (default: 500_000)")
    parser.add_argument("--N_EPOCHS", type=int, default=200, 
                        help="Number of EPOCHS (default: 100)")
    parser.add_argument("--seed", type = int, default = 1,
                        help = "See number (default: 1)")
    parser.add_argument("--layer_len", type = int, default = 256,
                        help = "layer length of FL network (default: 256)")
    parser.add_argument('--calibrate', action='store_true', 
                        help="calibrate or not (default: False)")
    parser.add_argument("--num_calibrations", type=int, default=10_000_000,
                        help="Number of calibrations for sampling (default: 10_000_000)")
    parser.add_argument("--iter_calibrations", type=int, default=40,
                        help="Number of iterations for calibrations (default: 40)")
    parser.add_argument('--c2st', action='store_true', 
                        help="Caculate c2st after calibrating (default: False)")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
    #main_cond(args)
    
    # Use the parsed arguments
    print(f"task: {args.task}")
    print(f"Number of simulations: {args.num_training}")
    print(f"Number of epochs: {args.N_EPOCHS}")
    print(f"seed: {args.seed}")