import torch
import numpy as np
import argparse
import os
import subprocess
from module import FL_Net, GRU_net
import time
from NCoinJDP import NCoinJDP_train, ABC_rej
from simulator import Simulators, Priors, get_task_parameters
#from utils.batch_process import resid_chunk_process

# Set the default device based on availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(args):
    # Set seeds
    torch.set_default_device("cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)   

    # Initialize the Priors and Simulators classes
    priors = Priors(args.task)
    
    task_params = get_task_parameters(args.task)
    limits = task_params["limits"]
    n = task_params["n"]
    delta = task_params["delta"]

    # Sample theta from the prior
    theta = priors().sample((args.num_training,))

    # Run the simulator
    simulators = Simulators(args.task, n = n, delta = delta)
    X = simulators(theta)
    
    # Learning hyperparameters
    D_in, D_out, Hs = X.size(1), theta.size(1), args.layer_len

    # Save the models
    ## Define the output directory
    print(f"start", flush=True)
    output_dir = f"../../depot_hyun/hyun/NCoinJDP/{args.experiment}/{args.task}/J_{int(args.num_training/1000)}K"
    #output_dir = "../depot_hyun/NABC_nets_RAdam/" + args.task
    ## Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created.")
    else:
        print(f"Directory '{output_dir}' already exists.")
    
    if args.task == "MROUJ" or args.task == "OU":
        net = GRU_net(input_dim = 1, hidden_dim = Hs, output_dim = D_out)
    else:
        net = FL_Net(D_in, D_out, H=Hs, H2=Hs, H3=Hs).to(device)
        
    # Train Mean Function
    print(f"start training for mean function", flush=True)
    start_time = time.time()  # Start timer
    val_batch = 1_000 if args.task == "OU" else 10_000
    tmp, best_error = NCoinJDP_train(X, theta, net, device=device, N_EPOCHS=args.N_EPOCHS, val_batch = val_batch)
    end_time = time.time()
    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f"Mean Function Training completed in {elapsed_time/60:.2f} mins")
    
    net.load_state_dict(tmp)

    torch.save(net.state_dict(),  output_dir + "/" + args.task + str(args.seed) +"_mean.pt")
    torch.save([elapsed_time, best_error, torch.cuda.get_device_name(0)],  output_dir + "/" + args.task + str(args.seed) +"_info.pt")
    
    net = net.to("cpu")
    print("## cMAD training job script submitted ##")

def main_cond(args):
    create_job_script(args)
    print("## cMAD training job script submitted ##")
        

def create_job_script(args):
    calibrate_flag = "--calibrate" if args.calibrate else ""
    job_script = f"""#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --account=standby
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --output=output_log/{args.task}/output_log_%A.log
#SBATCH --error=output_log/{args.task}/error_log_%A.txt

# Create the output_log directory if it doesn't exist
mkdir -p output_log/{args.task}

# Load the required Python environment
module use /depot/wangxiao/etc/modules
module load conda-env/sbi_pack-py3.11.7

# Change to the directory where the job was submitted from
SLURM_SUBMIT_DIR=$(pwd)
cd $SLURM_SUBMIT_DIR

# Run the Python script for the current simulation
echo "Running simulation for task {args.task}, num_training: {args.num_training}, N_EPOCHS: {args.N_EPOCHS} seed={args.seed}, layer_len={args.layer_len}..."
python NABC_training_cMAD --task {args.task} --num_training {args.num_training} --N_EPOCHS {args.N_EPOCHS} --seed {args.seed} --layer_len {args.layer_len} {calibrate_flag} --num_calibrations {args.num_calibrations} --iter_calibrations {args.iter_calibrations}
"""
    # Create the directory for SLURM files if it doesn't exist
    output_dir = f"../depot_hyun/NABC_nets/{args.task}/J_{int(args.num_training/1000)}K/slurm_files"
    os.makedirs(output_dir, exist_ok=True)

    job_file_path = os.path.join(output_dir, f"cMAD_{args.task}_{args.num_training}_{args.seed}.sh")
    with open(job_file_path, 'w') as f:
        f.write(job_script)
    print(f"Job script created: {job_file_path}")

    # Submit the job immediately
    subprocess.run(['sbatch', job_file_path])
    print(f"Job {job_file_path} submitted.")


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