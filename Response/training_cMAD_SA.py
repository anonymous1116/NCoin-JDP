import torch
import numpy as np
import argparse
import os
import argparse
import subprocess
from module import FL_Net
import time
from simulator import Simulators, Priors, get_task_parameters
from NCoinJDP import cond_mad_train, ABC_rej
from utils.batch_process import resid_chunk_process

# Set the default device based on availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(args):
    # Set seeds
    torch.set_default_device("cpu")
    torch.manual_seed(args.seed * 2)
    np.random.seed(args.seed * 2)   

    # Initialize the Priors and Simulators classes
    priors = Priors(args.task)
    
    task_params = get_task_parameters(args.task)
    limits = task_params["limits"]
    n = task_params["n"]
    delta = task_params["delta"]
    x0_list = task_params["x0_list"]
    x0 = torch.tensor([x0_list], dtype=torch.float32).to("cpu")
    print(f"x0: {x0}")
    
    # Sample theta from the prior
    theta = priors().sample((args.num_training*10,))
    
    # Run the simulator
    simulators = Simulators(args.task, n = n, delta = delta)
    X = simulators(theta)
    
    X, theta = ABC_rej(x0, X, theta, tol=0.1, device=device)
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
    tmp = cond_mad_train(X, resid, net_var, device=device, N_EPOCHS=args.N_EPOCHS, val_batch = val_batch)

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
    parser.add_argument('--experiment', type=str, default='SA1', 
                        help='experiment type: S1 ...')
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
    

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
    print(f"task: {args.task}")
    print(f"Number of simulations: {args.num_training}")
    print(f"Number of epochs: {args.N_EPOCHS}")
    print(f"seed: {args.seed}")