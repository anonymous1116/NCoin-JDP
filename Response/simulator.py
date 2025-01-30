import torch
import os
import numpy as np
import torch.distributions as D
from sbi.utils import BoxUniform

def get_task_parameters(task):
    x0_list = []
    if task == "OU_summary" or task == "OU":
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the full path to the slcp2 file
        file_path = os.path.join(current_dir, "OU_obs.pt")
        tmp = torch.load(file_path)
        
        x0_list = tmp[0].numpy().tolist()
    task_params = {"OU_summary": {"x0_list": x0_list if x0_list else [],  
                 "limits": [[1, 5], [1, 2.5], [0.5, 2.0]],
                 "n": int(500),
                 "delta": 1/12
                }
    }
    if task not in task_params:
        raise ValueError(f"Unknown task: {task}")
    return task_params[task]

class Priors:
    def __init__(self, task):
        self.task = task

    def __call__(self):
        # Call the appropriate prior function based on the task
        if self.task == 'OU':
            return self.OU()
        if self.task == 'OU_summary':
            return self.OU()
        
        
    def OU(self):
        return BoxUniform(low=torch.tensor([1, 1, 0.5]), high=torch.tensor([5, 2.5, 2]))


class Simulators:
    def __init__(self, task, n, delta):
        self.task = task
        self.n = n
        self.delta = delta

    def __call__(self, theta):
            if self.task == "OU":
                return self.OU(theta)
            elif self.task =="OU_summary":
                return self.OU_summary(self.OU(theta))
            
    def OU(self, theta):
        L_OU = theta.size(0)
        time_OU = np.arange(0,self.n+1)/self.n * self.n * self.delta
        mu_OU, theta_OU, sigma2_OU = theta[:,0], theta[:, 1], theta[:, 2]
        z0 = torch.normal(theta_OU, torch.sqrt(sigma2_OU/(2*mu_OU)))
        path_OU = torch.zeros(L_OU, time_OU.size)
        path_OU[:,0] = z0
        for l in range(time_OU.size-1):
            del_L = time_OU[l+1] - time_OU[l]
            OU_mean = z0 * torch.exp(-mu_OU * del_L) + theta_OU * (1- torch.exp(-mu_OU * del_L))
            OU_sd = torch.sqrt( sigma2_OU/(2*mu_OU) * (1- torch.exp(-2 * mu_OU * del_L)) )
            z0 = torch.normal(OU_mean, OU_sd)
            path_OU[:,l+1] = z0
        return(path_OU)
    
    def OU_summary(self, X):
        """
        X: torch size: [L,n]
        """
        n0 = X.size()[1]

        # Efficient vectorized computation
        X_prev = X[:, :-1]  # X_{i-1}
        X_next = X[:, 1:]   # X_{i}

        sum1 = torch.sum(X_next * X_prev, dim=1)
        sum2 = torch.sum(X_next, dim=1)
        sum3 = torch.sum(X_prev, dim=1)
        sum4 = torch.sum(X_prev**2, dim=1)
        sum5 = torch.sum(X_next**2, dim=1)

        n0 = X.size(1)

        # Compute summary statistics
        S1 = (sum1 - sum2 * sum3 / n0) / n0
        S2 = sum2 / n0
        S3 = sum3 / n0
        S4 = sum4 / n0 - (sum3 / n0)**2
        S5 = sum5 / n0 - (sum2 / n0)**2

        return torch.stack((S1, S2, S3, S4, S5), dim=1)