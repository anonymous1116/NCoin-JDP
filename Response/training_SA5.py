import torch
import numpy as np
import argparse
import os
import subprocess
from module import FL_Net2
from sbi.utils import BoxUniform

from NCoinJDP import NCoinJDP_train, ABC_rej
from simulator import Simulators, PBJD_truncated_priors, PBJD_theta_log_transform
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
    
    test_save_name = 'Robustness/rda_data.pt'
    test_data= torch.load(test_save_name)
    x0=test_data[args.x0_ind]
    x0 = x0.type(torch.float32)


    simulators = Simulators("PBJD_summary", n = n, delta = delta)
    
    param = [[-0.01, 0.02], [100], [1], [1], [1/100], [1/100]]
    trunc = [None, None, None, None, None, None]

    # Initialize the Priors and Simulators classes
    if args.priors == "P1_0":
        param[0] = [-0.01, 0.01]
    elif args.priors == "P1_1":
        param[0] = [-0.01, 0.02]
    elif args.priors == "P1_2":
        param[0] = [-0.02, 0.02]
    elif args.priors == "P1_3":
        param[0] = [-0.02, 0.03]
    elif args.priors == "P1_4":
        param[0] = [-0.03, 0.03]

    theta = PBJD_truncated_priors(args.num_training, param, trunc)
    print(f"Prior generated", flush=True)
    
    X = simulators(theta)
    print(f"Samples generated", flush=True)
    
    D_in, D_out, Hs = X.size(1), theta.size(1), args.layer_len

    #cases = args.priors[:,2]
    output_dir = f"../../depot_hyun/hyun/NCoinJDP/{args.experiment}/{args.task}/J_{int(args.num_training/1000)}/{args.priors}/C{args.x0_ind}"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created.")
    else:
        print(f"Directory '{output_dir}' already exists.")

    x0 = torch.reshape(x0, (1, x0.size(0)))
    x0 = simulators.PBJD_summary(x0)

    tol = .05
    print(f"ABC_rej start", flush=True)
    
    tmp = ABC_rej(x0, X, theta, tol = tol, device = device)
    
    trunc = []
    for j in range(6):
        temp = np.array([torch.min(tmp[1],0).values.tolist(),torch.max(tmp[1],0).values.tolist()])[:,j].tolist()
        trunc.append(temp)

    
    theta_new =  PBJD_truncated_priors(args.num_training, param, trunc)
    print(f"Prior_new generated", flush=True)
    
    X_new = simulators(theta_new)
    print(f"X_new generated", flush=True)
    
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
    tmp, _ = NCoinJDP_train(X_new, theta_transform, net, device=device, N_EPOCHS=args.N_EPOCHS, val_batch = val_batch, l2 = "True")
    net.load_state_dict(tmp)
    net.eval()
    net.to("cpu")

    print(f"saving", flush=True)
    torch.save([net(x0).detach(),a,b], f"{output_dir}/local_{args.seed}.pt")
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
    parser.add_argument("--priors", type = str, default = "P0",
                        help = "priors")
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