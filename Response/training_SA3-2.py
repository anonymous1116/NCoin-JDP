import torch
import numpy as np
import argparse
import os
import subprocess
from module import FL_Net, GRU_net
from sbi.utils import BoxUniform

from NCoinJDP import NCoinJDP_train, ABC_rej
from simulator import Simulators
#from utils.batch_process import resid_chunk_process

# Set the default device based on availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(args):
    # Set seeds
    torch.set_default_device("cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)   
    n = 3000
    delta = 1/52
    
    test_save_name = '../../depot_hyun/hyun/test_data/OU_test_n'+ str(n) + '_' + "S1" +'.pt'
    test_data= torch.load(test_save_name)
    my_test=test_data[0]

    simulators = Simulators(args.task, n = n, delta = delta)
    
    # Initialize the Priors and Simulators classes
    if args.priors == "P1":
        ub = 5.5
    elif args.priors == "P2":
        ub = 6
    elif args.priors == "P3":
        ub = 6.5
    elif args.priors == "P4":
        ub = 7
    else:
        ub = 5
    priors = BoxUniform(low=torch.tensor([1, 1, 0.5]), high=torch.tensor([ub, 2.5, 2]))
    theta = priors.sample((args.num_training,))

    
    print(f"start", flush=True)
    
    X = simulators(theta)
    D_in, D_out, Hs = X.size(1), theta.size(1), args.layer_len

    for j in range(4):
        output_dir = f"../../depot_hyun/hyun/NCoinJDP/{args.experiment}/{args.task}/J_{int(args.num_training/1000)}/{args.priors}/C{j}"
    
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Directory '{output_dir}' created.")
        else:
            print(f"Directory '{output_dir}' already exists.")

        x0 = my_test[j][args.seed]
        x0 = torch.reshape(x0, (1, x0.size(0)))
        x0 = simulators.OU_summary(x0)

        tol = .05
        print(f"ABC_rej start", flush=True)
    
        tmp = ABC_rej(x0, X, theta, tol = tol, device = device)
    
        priors_new = BoxUniform(low=torch.tensor(torch.min(tmp[1],0).values.tolist()), high=torch.tensor(torch.max(tmp[1],0).values.tolist()))
        theta_new = priors_new.sample((100_000,))
        X_new = simulators(theta_new)

        print(f"training start", flush=True)
    
        net = FL_Net(D_in, D_out, H=Hs, H2=Hs, H3=Hs).to(device)
        val_batch = 10000
        tmp, _ = NCoinJDP_train(X_new, theta_new, net, device=device, N_EPOCHS=args.N_EPOCHS, val_batch = val_batch, l2 = "True")
        net.load_state_dict(tmp)
        net.eval()
        net.to("cpu")

        print(f"saving", flush=True)

        torch.save(net(x0).detach(), f"{output_dir}/local_{args.seed}.pt")
        del X_new, theta_new, priors_new, net
        torch.set_default_device("cpu")
    
    
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