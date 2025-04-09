import torch
import numpy as np
import argparse
import os
import sys
import copy
import subprocess
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from NCoinJDP import NCoinJDP_train, ABC_rej, learning_checking_save
from simulator import Simulators, PBJD_theta_exp_transform, PBJD_theta_log_transform, PBJD_truncated_priors2
#from utils.batch_process import resid_chunk_process

# Set the default device based on availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(args):
    # Set seeds
    torch.set_default_device("cpu")
    torch.manual_seed(12345+args.seed)
    np.random.seed(12345+args.seed)

    n = 2014
    delta = 1

    simulators = Simulators("PBJD_summary", n = n, delta = delta)


    param = [[-0.01, 0.02], [100], [0.05, 2], [0.05,2], [1/100], [1/100]]
    trunc = [[-0.01, 0.02], [1e-5, 1e-2], [0.05, 2], [0.05, 2], [10, 300], [10, 300]]


    # Initialize the Priors and Simulators classes
    if args.priors == "P1_0":
        param[0] = [-0.01, 0.02]
        trunc[0] = [-0.01, 0.02]
    elif args.priors == "P1_1":
        param[0] = [-0.01, 0.025]
        trunc[0] = [-0.01, 0.025]
    elif args.priors == "P1_2":
        param[0] = [-0.015, 0.025]
        trunc[0] = [-0.015, 0.025]
    elif args.priors == "P1_3":
        param[0] = [-0.015, 0.03]
        trunc[0] = [-0.015, 0.03]
    elif args.priors == "P1_4":
        param[0] = [-0.020, 0.03]
        trunc[0] = [-0.020, 0.03]

    if args.priors == "P2_0":
        param[1] = [100]
        trunc[1] = [1e-5, 1e-2]
    elif args.priors == "P2_1":
        param[1] = [75]
        trunc[1] = [1e-5, 5e-2]
    elif args.priors == "P2_2":
        param[1] = [50]
        trunc[1] = [1e-5, 1e-2]
    elif args.priors == "P2_3":
        param[1] = [100]
        trunc[1] = [1e-5, 3e-2]
    elif args.priors == "P2_4":
        param[1] = [100]
        trunc[1] = [1e-5, 5e-2]

    if args.priors == "P3_0":
        param[2] = [0.05, 2.0]
        trunc[2] = [0.05, 2.0]
    elif args.priors == "P3_1":
        param[2] = [0.05, 2.5]
        trunc[2] = [0.05, 2.5]
    elif args.priors == "P3_2":
        param[2] = [0.05, 3.0]
        trunc[2] = [0.05, 3.0]
    elif args.priors == "P3_3":
        param[2] = [0.05, 3.5]
        trunc[2] = [0.05, 3.5]
    elif args.priors == "P3_4":
        param[2] = [0.05, 4.0]
        trunc[2] = [0.05, 4.0]

    if args.priors == "P4_0":
        param[4] = [1/100]
        trunc[4] = [10, 300]
    elif args.priors == "P4_1":
        param[4] = [1/150]
        trunc[4] = [10, 300]
    elif args.priors == "P4_2":
        param[4] = [1/200]
        trunc[4] = [10, 300]
    elif args.priors == "P4_3":
        param[4] = [1/100]
        trunc[4] = [10, 350]
    elif args.priors == "P4_4":
        param[4] = [1/100]
        trunc[4] = [10, 400]

    num_calibrations = 5_000

    # Run the simulator
    theta_cal = PBJD_truncated_priors2(num_calibrations, param, trunc).to(device)
    X_cal = simulators(theta_cal)

    output_dir = f"scratch/gilbreth/hyun18/PBJD/{args.priors}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created.")
    else:
        print(f"Directory '{output_dir}' already exists.")
    
    torch.save([X_cal, theta_cal], f"{output_dir}/{args.priors}_{args.seed}")


def get_args():
    parser = argparse.ArgumentParser(description="Run simulation with customizable parameters.")
    parser.add_argument("--seed", type = int, default = 1,
                        help = "See number (default: 1)")
    parser.add_argument("--priors", type = str, default = "P1_0",
                        help = "priors")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
    #main_cond(args)
    
    # Use the parsed arguments
    print(f"seed: {args.seed}")
    print(f"priors: {args.priors}")
    