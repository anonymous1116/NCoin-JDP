import torch
import numpy as np
import argparse
import os
import subprocess
from module import FL_Net, GRU_net
from sbi.utils import BoxUniform

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
    if args.priors == "P1":
        ub = 6
    elif args.priors == "P2":
        ub = 7
    elif args.priors == "P3":
        ub = 8
    elif args.priors == "P4":
        ub = 9
        
    priors = BoxUniform(low=torch.tensor([1, 1, 0.5]), high=torch.tensor([ub, 2.5, 2]))
    
    #task_params = get_task_parameters(args.task)
    n = 3000
    delta = 1/52
    
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
    output_dir = f"../../depot_hyun/hyun/NCoinJDP/{args.experiment}/{args.task}/{args.priors}"
    
    #output_dir = "../depot_hyun/NABC_nets_RAdam/" + args.task
    ## Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created.")
    else:
        print(f"Directory '{output_dir}' already exists.")
    
    if args.task == "MROUJ" or args.task == "OU" or args.task == "CIR":
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
    parser.add_argument("--priors", type = int, default = "P0",
                        help = "priors")
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