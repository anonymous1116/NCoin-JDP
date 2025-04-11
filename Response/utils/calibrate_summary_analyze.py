import torch
import numpy as np
import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from simulator import PBJD_theta_log_transform, PBJD_theta_exp_transform
from NCoinJDP import ABC_rej, calibrate_PBJD, quantile_print, compute_mad
from module import FL_Net2

def main(args):
    # Set seeds
    torch.set_default_device("cpu")
    device = torch.device("cuda:0")

    # x0_read
    test_save_name = 'Robustness/RDA_data.pt'
    test_data= torch.load(test_save_name)
    x0s = test_data.type(torch.float32)

    j = args.x0_ind
    x0 = torch.clone(x0s[j])
    x0 = torch.tensor(np.array(x0),dtype = torch.float32)
    x0 = torch.reshape(x0, (1, x0.size(0)))

    # Calibration data read
    X_new = []
    theta_new = []
    for k in range(1, 101):
        data_dir = f"/scratch/gilbreth/hyun18/PBJD/{args.priors}/{args.priors}_{k}"
        [X_cal_tmp, theta_cal_tmp] = torch.load(data_dir)
        X_new_tmp, theta_new_tmp = ABC_rej(x0, X_cal_tmp, theta_cal_tmp, tol = .1, device =device)
        X_new.append(X_new_tmp)
        theta_new.append(theta_new_tmp)
    X_new = torch.cat(X_new,dim = 0)
    theta_new = torch.cat(theta_new, dim=0)
    print(X_new.size(), flush = True)


    D_in, D_out, Hs = X_new.size(1), theta_new.size(1), 512
    tol = np.arange(0.1, 1.1 ,0.1)
    output_dir = f"../../depot_hyun/hyun/NCoinJDP/SA6_analyze/PBJD_summary/J_2000/C{j}"
        
    for seed in range(1, 11, 1):
        net_tmp = torch.load(f"{output_dir}/mean_nets_{seed}.pt")
        _, a, b=  torch.load(f"{output_dir}/local_{seed}.pt")
        net_tmp2 = torch.load(f"{output_dir}/cond_nets_{seed}.pt")

        net = FL_Net2(D_in, D_out, H=Hs, H2=Hs, H3=Hs)
        net_var = FL_Net2(D_in, D_out, H=Hs, H2=Hs, H3=Hs)
        net.load_state_dict(net_tmp)
        net_var.load_state_dict(net_tmp2)
        net.eval()
        net_var.eval()

        X_new_x0 = torch.clone( (X_new - a) / (b -a ))
        x0_new = torch.clone((x0-a)/(b-a))

        theta_transform_new_x0 = torch.clone(PBJD_theta_log_transform(theta_new))

        theta_samples = []
        for k in range(10):
            tmp, _ = calibrate_PBJD(x0_new, X_new_x0, theta_transform_new_x0, net, net_var, n_samples = 10_000, device =device, tol=tol[k])
            theta_samples.append(tmp)
        theta_samples = torch.cat(theta_samples, dim =0)
        theta_samples = PBJD_theta_exp_transform(theta_samples)

        q_results_95 = quantile_print(theta_samples, alpha = .05)
        q_results_90 = quantile_print(theta_samples, alpha = .10)
        q_results_85 = quantile_print(theta_samples, alpha = .15)

        mad_results = compute_mad(theta_samples)

        if not os.path.exists(f"{output_dir}"):
            os.makedirs(f"{output_dir}")
            print(f"Directory '{output_dir}' created.")
        else:
            print(f"Directory '{output_dir}' already exists.")
        

        torch.save(q_results_95, f"{output_dir}/q_results_95_{seed}")
        torch.save(q_results_90, f"{output_dir}/q_results_90_{seed}")
        torch.save(q_results_85, f"{output_dir}/q_results_85_{seed}")
        
        torch.save(mad_results, f"{output_dir}/mad_results_{seed}")
        
        del net, net_var, X_new_x0, theta_transform_new_x0, theta_samples, x0_new


def get_args():
    parser = argparse.ArgumentParser(description="Run simulation with customizable parameters.")
    parser.add_argument("--x0_ind", type = int, default = 0,
                        help = "x0_ind (default: 0)")
    parser.add_argument("--priors", type = str, default = "P1_0",
                        help = "priors")
    parser.add_argument('--experiment', type=str, default='SA1', 
                        help='experiment type: S1 ...')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(args)
