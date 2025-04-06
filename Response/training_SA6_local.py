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
    x0=test_data[args.x0_ind]
    x0 = x0.type(torch.float32)

    x0 = torch.reshape(x0, (1, x0.size(0)))
    x0 = simulators.PBJD_summary(x0)

    
    param = [[-0.01, 0.02], [100], [0.05, 2], [0.05,2], [1/100], [1/100]] 
    trunc = [[-0.01, 0.02], [1e-5, 1e-2], [0.05, 2], [0.05, 2], [10, 300], [10, 300] ]

    print(f"ABC_rej start", flush=True)
    
    all_theta = []
    all_X = []

    batch_size = 200_000
    total_samples = args.num_training
    tol = args.tol
    
    for start in range(0, total_samples, batch_size):
        end = min(start + batch_size, total_samples)
        current_batch_size = end - start

        theta_batch = PBJD_truncated_priors2(current_batch_size, param, trunc)
        theta_batch = theta_batch.to(device)

        with torch.no_grad():
            X_batch = simulators(theta_batch)
        X_new_batch, theta_new_batch = ABC_rej(x0, X_batch, theta_batch, tol = tol, device = device)
        all_theta.append(theta_new_batch.cpu())
        all_X.append(X_new_batch.cpu())

        del theta_batch, X_batch
        torch.cuda.empty_cache()

    theta_new = torch.cat(all_theta, dim=0)
    X_new = torch.cat(all_X, dim=0)

    print(f"Samples generated", flush=True)
    
    D_in, D_out, Hs = X_new.size(1), theta_new.size(1), args.layer_len

    #cases = args.priors[:,2]
    output_dir = f"../../depot_hyun/hyun/NCoinJDP/{args.experiment}/{args.task}/J_{int(args.num_training/1000)}/C{args.x0_ind}"
    
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
    torch.save([net(x0).detach(),a,b], f"{output_dir}/local_{args.seed}.pt")
    learning_checking_save(X_new, theta_transform, net, name = f"{output_dir}/LC_local_{args.seed}.pdf")
    print(f"{args.experiment}, {args.task}, seed {args.seed}, priors {args.priors}, x0_ind, {args.x0_ind} done", flush=True)
    
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