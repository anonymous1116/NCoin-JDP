import torch
import os
import numpy as np
import torch.distributions as D
from sbi.utils import BoxUniform

def get_task_parameters(task):
    task_params = {"OU_summary": {"x0_list": x0_list if x0_list else [],  
                 "limits": [[1, 5], [1, 2.5], [0.5, 2.0]],
                 "n": int(500),
                 "delta": 1/12
                }
    }

class Priors:
    def __init__(self, task):
        self.task = task

    def __call__(self):
        # Call the appropriate prior function based on the task
        if self.task == 'OU':
            return self.OU()
        
    def OU(self):
        return BoxUniform(low=torch.tensor([1, 1, 0.5]), high=torch.tensor([5, 2.5, 2]))


class Simulators:
    def __init__(self, task, n, delta):
        self.task = task
        self.n = n
        self.delta = delta

    def __call__(self, theta, n, delta):
            if self.task == "OU":
                return self.OU(theta, n, delta)
            elif self.task =="OU_summary":
                return self.OU_summary(self.OU(theta, n, delta))
            
    def OU(self, theta, n, delta):
        L_OU = theta.size(0)
        time_OU = np.arange(0,n+1)/n * n * delta
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
        L0 = X.size()[0]
        n0 = X.size()[1]

        sum1 = torch.zeros(L0) # sum x_i x_{i-1}
        sum2 = torch.zeros(L0) # sum x_i
        sum3 = torch.zeros(L0) # sum x_{i-1}
        sum4 = torch.zeros(L0) # sum x_{i-1}^2
        sum5 = torch.zeros(L0) # sum x_{i}^2

        for l in range(n0-1):
            sum1 = sum1 + X[:,l+1] * X[:,l]
            sum2 = sum2 + X[:,l+1]
            sum3 = sum3 + X[:,l]
            sum4 = sum4 + torch.pow(X[:,l],2)
            sum5 = sum5 + torch.pow(X[:,l+1],2)
        return(torch.stack(( (sum1 - sum2 * sum3/n0 ) /n0, sum2/n0, sum3/n0, sum4/n0 - (sum3/n0)**2, sum5/n0 - (sum2/n0)**2) ,1))