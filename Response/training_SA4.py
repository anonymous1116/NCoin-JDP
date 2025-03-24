import torch
import numpy as np
import argparse
import os
import subprocess
from module import FL_Net, GRU_net
from torch.distributions import Exponential

import time
from NCoinJDP import NCoinJDP_train
from simulator import Simulators, truncated_normal
#from utils.batch_process import resid_chunk_process

# Set the default device based on availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(args):
    # Set seeds
    torch.set_default_device("cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)   

    n = 2015
    delta = 1
    
    beta_range = [-0.01, 0.02]
    sigma_param = [100]

    lamb_p_param = [1]
    lamb_n_param = [1]

    eta_p_param = [100]
    eta_n_param = [100]

    # Training + validation + Test data generating
    torch.manual_seed(510)

    L = args.num_training
    beta_ran  = torch.rand(L) * (beta_range[1] - beta_range[0]) + beta_range[0]
    sigma_ran = Exponential(sigma_param[0] * torch.ones(L)).sample()

    lamb_p_ran = Exponential(lamb_p_param[0] * torch.ones(L)).sample()
    lamb_n_ran = Exponential(lamb_n_param[0] * torch.ones(L)).sample()

    eta_p_ran = Exponential(eta_p_param[0] * torch.ones(L)).sample()
    eta_n_ran = Exponential(eta_n_param[0] * torch.ones(L)).sample()

    
    theta_raw = torch.stack((beta_ran, sigma_ran, lamb_p_ran, lamb_n_ran, eta_p_ran, eta_n_ran), dim = 1)
    theta_transform = torch.stack((beta_ran, torch.log(sigma_ran), torch.log(lamb_p_ran), torch.log(lamb_n_ran), 
                                    torch.log(eta_p_ran), torch.log(eta_n_ran)), dim = 1)
        

    # Initialize the Priors and Simulators classes
    if args.priors == "P1_1":
        del beta_ran
        beta_range = [-0.02, 0.04]
        beta_ran  = torch.rand(L) * (beta_range[1] - beta_range[0]) + beta_range[0]
    elif args.priors == "P1_2":
        del beta_ran
        beta_range = [-0.03, 0.1]
        beta_ran  = torch.rand(L) * (beta_range[1] - beta_range[0]) + beta_range[0]
    elif args.priors == "P1_3":
        del beta_ran
        beta_range = [-0.06, 0.12]
        beta_ran  = torch.rand(L) * (beta_range[1] - beta_range[0]) + beta_range[0]
    elif args.priors == "P1_4":
        del sigma_ran
        sigma_ran = Exponential(200.0 * torch.ones(L)).sample()
    elif args.priors == "P1_5":
        del sigma_ran
        sigma_ran = Exponential(50.0 * torch.ones(L)).sample()
    elif args.priors == "P1_6":
        del sigma_ran
        sigma_ran = Exponential(20.0 * torch.ones(L)).sample()
    
    theta_raw = torch.stack((beta_ran, sigma_ran, lamb_p_ran, lamb_n_ran, eta_p_ran, eta_n_ran), dim = 1)
    

    # Run the simulator
    simulators = Simulators(args.task, n = n, delta = delta)
    X_raw = simulators(theta_raw)
    
    theta_transform = torch.stack((beta_ran, torch.log(sigma_ran), torch.log(lamb_p_ran), torch.log(lamb_n_ran), 
                                    torch.log(eta_p_ran), torch.log(eta_n_ran)), dim = 1)
    

    a = torch.quantile(X_raw, .001, 0)
    a = torch.reshape(a, (1, a.size()[0]))
    b = torch.quantile(X_raw, .999, 0)
    b = torch.reshape(b, (1, b.size()[0]))

    X = torch.clone((X_raw - a) / (b - a))
    
    # Learning hyperparameters
    D_in, D_out, Hs = X.size(1), theta_transform.size(1), args.layer_len

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
    tmp, best_error = NCoinJDP_train(X, theta_transform, net, device=device, N_EPOCHS=args.N_EPOCHS, val_batch = val_batch)
    end_time = time.time()
    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f"Mean Function Training completed in {elapsed_time/60:.2f} mins")
    
    net.load_state_dict(tmp)

    torch.save([net.state_dict(), a, b],  output_dir + "/" + args.task + str(args.seed) +"_mean.pt")
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
    parser.add_argument("--priors", type = str, default = "P1_0",
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