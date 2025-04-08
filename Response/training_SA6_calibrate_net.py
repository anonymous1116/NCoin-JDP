import torch
import numpy as np
import argparse
import os
import subprocess
from module import FL_Net2
from sbi.utils import BoxUniform

from NCoinJDP import NCoinJDP_train, ABC_rej, learning_checking_save
from simulator import Simulators, PBJD_theta_exp_transform, PBJD_theta_log_transform, PBJD_truncated_priors2
#from utils.batch_process import resid_chunk_process

# Set the default device based on availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(args):
    # Set seeds
    torch.set_default_device("cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    n = 2014
    delta = 1
    simulators = Simulators("PBJD_summary", n = n, delta = delta)
    
    test_save_name = 'Robustness/RDA_data.pt'
    test_data= torch.load(test_save_name)
    x0= test_data[args.x0_ind]
    
    x0 = torch.reshape(x0, (1, x0.size(0)))
    
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

    print(f"ABC_rej start", flush=True)
    batch_size = 200_000
    total_samples = args.num_training
    tol = args.tol    
    X_new, theta_new = batched_ABC_simulation(total_samples, batch_size, param, trunc, x0, simulators, tol, device)
    print(f"Samples generated", flush=True)
    
    D_in, D_out, Hs = X_new.size(1), theta_new.size(1), args.layer_len
    #cases = args.priors[:,2]
    output_dir = f"../../depot_hyun/hyun/NCoinJDP/{args.experiment}/{args.task}/J_{int(args.num_training/1000)}/{args.priors}"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created.")
    else:
        print(f"Directory '{output_dir}' already exists.")
    
    theta_transform = PBJD_theta_log_transform(theta_new)
    a = torch.quantile(X_new, .001, 0)
    a = torch.reshape(a, (1, a.size()[0]))
    b = torch.quantile(X_new, .999, 0)
    b = torch.reshape(b, (1, b.size()[0]))

    X_new = torch.clone((X_new - a) / (b - a))
    x0 = torch.clone((x0 - a) / (b - a))

    print(f"training start", flush=True)
    net = FL_Net2(D_in, D_out, H=Hs, H2=Hs, H3=Hs).to(device)
    val_batch = 10000
    tmp, _ = NCoinJDP_train(X_new, theta_transform, net, device=device, N_EPOCHS=args.N_EPOCHS, val_batch = val_batch)
    net.load_state_dict(tmp)
    net.eval()
    net.to("cpu")

    print(f"saving", flush=True)
    torch.save(net, f"{output_dir}/mean_nets_{args.seed}.pt")
    torch.save([net(x0).detach(),a,b], f"{output_dir}/local_{args.seed}.pt")
    learning_checking_save(X_new, theta_transform, net, name = f"{output_dir}/LC_local_{args.seed}.pdf")
    print(f"{args.experiment}, {args.task}, seed {args.seed}, priors {args.priors}, x0_ind, {args.x0_ind} done", flush=True)
    
    del X_new, theta_new
    torch.manual_seed(args.seed*2)
    np.random.seed(args.seed*2)
    torch.set_default_device("cpu")
       
    X_new, theta_new = batched_ABC_simulation(total_samples, batch_size, param, trunc, x0, simulators, tol, device)
    
    X_new = torch.clone((X_new - a) / (b - a))
    theta_transform = PBJD_theta_log_transform(theta_new)
    
    net.eval()
    torch.set_default_device("cpu")
        
    resid  = theta_new - net.to("cpu")(X_new).detach()
    print(resid)
    resid = torch.max(torch.abs(resid), torch.ones(1) * 1e-30).log()

    net_var = FL_Net2(D_in, D_out, H=Hs, H2=Hs, H3=Hs).to(device)

    tmp, _ = NCoinJDP_train(X_new, resid, net_var, device=device, N_EPOCHS=args.N_EPOCHS, val_batch = val_batch)
    net_var.load_state_dict(tmp)
    net_var.eval()
    net_var.to("cpu")
    torch.save(net_var, f"{output_dir}/cond_nets_{args.seed}.pt")
    

def batched_ABC_simulation(
    total_samples,
    batch_size,
    param,
    trunc,
    x0,
    simulators,
    tol=0.05,
    device="cuda"
):
    """
    Perform ABC rejection in batches to avoid memory issues.

    Args:
        total_samples (int): Total number of simulations to generate.
        batch_size (int): Number of samples per batch.
        param (list): Prior parameter settings for PBJD_truncated_priors2.
        trunc (list): Truncation bounds for the prior.
        x0 (Tensor): Observation to compare against.
        simulators (callable): Simulator function.
        ABC_rej (callable): ABC rejection function.
        tol (float): ABC tolerance.
        device (str): 'cuda' or 'cpu'.

    Returns:
        Tuple[Tensor, Tensor]: accepted_theta (N, D), accepted_X (N, ...).
    """
    all_theta = []
    all_X = []

    for start in range(0, total_samples, batch_size):
        end = min(start + batch_size, total_samples)
        current_batch_size = end - start

        # Sample from prior
        theta_batch = PBJD_truncated_priors2(current_batch_size, param, trunc).to(device)

        with torch.no_grad():
            X_batch = simulators(theta_batch)

        # Run ABC rejection
        X_new_batch, theta_new_batch = ABC_rej(x0, X_batch, theta_batch, tol=tol, device=device)

        # Store accepted samples
        all_theta.append(theta_new_batch.cpu())
        all_X.append(X_new_batch.cpu())

        del theta_batch, X_batch, X_new_batch, theta_new_batch
        torch.cuda.empty_cache()

    # Concatenate all accepted samples
    accepted_theta = torch.cat(all_theta, dim=0)
    accepted_X = torch.cat(all_X, dim=0)

    return accepted_X, accepted_theta



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
    parser.add_argument("--tol", type = float, default = 0.05,
                        help = "Tolerance value")
    parser.add_argument("--x0_ind", type = int, default = 0,
                        help = "x0_ind")
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
    