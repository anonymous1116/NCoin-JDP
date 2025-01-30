import torch
import numpy as np
import argparse
import os
import argparse
import subprocess
from module import FL_Net
import time
from simulator import Simulators, Priors, get_task_parameters
from utils.batch_process import resid_chunk_process

# Set the default device based on availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(args):
    # Set seeds
    torch.set_default_device("cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)   

    # Initialize the Priors and Simulators classes
    priors = Priors(args.task)
    simulators = Simulators(args.task)
    task_params = get_task_parameters(args.task)
    limits = task_params["limits"]

    # Sample theta from the prior
    theta = priors().sample((args.num_training,))

    # Run the simulator
    X = simulators(theta)
    
    # Learning hyperparameters
    D_in, D_out, Hs = X.size(1), theta.size(1), args.layer_len

    # Save the models
    ## Define the output directory
    print(f"cMAD training start", flush=True)
    output_dir = f"../../depot_hyun/hyun/NCoinJDP/{args.experiment}/{args.task}/J_{int(args.num_training/1000)}K"
    
    net = FL_Net(D_in, D_out, H=Hs, H2=Hs, H3=Hs).to("cpu")
    net_var = FL_Net(D_in, D_out, H=Hs, H2=Hs, H3=Hs).to("cpu")
    
    # Train Mean Function
    tmp = torch.load(output_dir + "/" + args.task + str(args.seed) +"_mean.pt")
    net.load_state_dict(tmp)

    # Conditional Deviation Function Learning
    resid = resid_chunk_process(X, theta, net, device = device, bounds = limits, chunk_size = 1_000 if args.task == "OU" else 10_000)
    
    resid = resid.detach().cpu()
    print(f"start training for conditional deviance function", flush=True)
    start_time = time.time()
    torch.manual_seed(args.seed * 2)
    val_batch = 1_000 if args.task == "OU" else 10_000
    tmp = cond_mad_train2(X, resid, net_var, device=device, N_EPOCHS=args.N_EPOCHS, val_batch = val_batch)

    end_time = time.time()
    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f"cMAD Training completed in {elapsed_time:.2f} seconds")
    
    
    net_var.load_state_dict(tmp)
    
    net = net.to("cpu")
    net_var = net_var.to("cpu")

    ## Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created.")
    else:
        print(f"Directory '{output_dir}' already exists.")
    torch.save([net.state_dict(),net_var.state_dict()] ,  output_dir + "/" + args.task + str(args.seed) +".pt")
    torch.save(elapsed_time, output_dir + "/" + args.task + str(args.seed) +"_time_cMAD.pt")
    torch.save(torch.cuda.get_device_name(0), output_dir + "/" + args.task + str(args.seed)+"_gpu_cMAD.pt")
    print("## DONE ##")



def get_args():
    parser = argparse.ArgumentParser(description="Run simulation with customizable parameters.")
    parser.add_argument('--task', type=str, default='twomoons', 
                        help='Simulation type: twomoons, MoG, MoUG, Lapl, GL_U or slcp, slcp2')
    parser.add_argument("--num_training", type=int, default=500_000,
                        help="Number of simulations for training (default: 500_000)")
    parser.add_argument("--N_EPOCHS", type=int, default=100, 
                        help="Number of EPOCHS (default: 100)")
    parser.add_argument("--seed", type = int, default = 1,
                        help = "See number (default: 1)")
    parser.add_argument("--layer_len", type = int, default = 256,
                        help = "layer length of FL network (default: 256)")
    parser.add_argument('--calibrate', action='store_true', 
                        help="calibrate or not (default: False)")
    parser.add_argument("--num_calibrations", type=int, default=10_000_000,
                        help="Number of calibrations for sampling (default: 100_000_000)")
    parser.add_argument("--iter_calibrations", type=int, default=40,
                        help="Number of iterations for calibrations (default: 40)")
    parser.add_argument('--c2st', action='store_true', 
                        help="Caculate c2st after calibrating (default: False)")
    

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
    if args.calibrate:
        # Build the command with the parsed arguments
        calibrate_command = [
            "python", "utils/calibrate_test.py",
            "--seed", str(args.seed),
            "--num_calibrations", str(args.num_calibrations),  # Replace with the appropriate variable if needed
            "--iter_calibrations", str(args.iter_calibrations),  # Replace with the appropriate variable if needed
            "--task", args.task,
            "--layer_len", str(args.layer_len),
            "--c2st",
            "--num_training", str(args.num_training),
            "--n_samples", "10000"
        ]

        # Execute the calibration script
        try:
            subprocess.run(calibrate_command, check=True)
            print(f"Calibration start with: {int(args.num_calibrations/1_000_000)}M_{int(args.iter_calibrations)} calibration sets")
        except subprocess.CalledProcessError as e:
            print(f"Calibration failed with error: {e}")

    # Print parsed arguments for verification
    print(f"task: {args.task}")
    print(f"Number of simulations: {args.num_training}")
    print(f"Number of epochs: {args.N_EPOCHS}")
    print(f"seed: {args.seed}")