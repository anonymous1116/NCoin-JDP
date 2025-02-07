import torch
import os
import numpy as np
import torch.distributions as D
from sbi.utils import BoxUniform

from torch.distributions.exponential import Exponential
from torch.distributions.beta import Beta
from torch.distributions.pareto import Pareto


def get_task_parameters(task):
    x0_list = []
    if task == "OU_summary" or task == "OU":
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the full path to the slcp2 file
        file_path = os.path.join(current_dir, "OU_obs_summary.pt")
        tmp = torch.load(file_path)
        
        x0_list = tmp[0].numpy().tolist()
    task_params = {"OU_summary": {"x0_list": x0_list if x0_list else [],  
                 "limits": [[1, 5], [1, 2.5], [0.5, 2.0]],
                 "n": int(501),
                 "delta": 1/12
                },
                "MROUJ_summary": {"x0_list": x0_list if x0_list else [],  
                 "limits": [[0.1, 3], [-1.0, 1.0], [0.1, 1.5], [0.01, 1], [0.1, 1.5]],
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
        elif self.task == 'OU_summary':
            return self.OU()
        elif self.task == 'MROUJ_summary':
            return self.MROUJ()
        
        
    def OU(self):
        return BoxUniform(low=torch.tensor([1, 1, 0.5]), high=torch.tensor([5, 2.5, 2]))

    def MROUJ(self):
            return BoxUniform(low=torch.tensor([0.1, -1, 0.1, 0.01, 0.1]), high=torch.tensor([3, 1, 1.5, 1, 1.5]))


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
            elif self.task =="MROUJ":
                return self.MROUJ(theta)
            elif self.task =="MROUJ_summary":
                return self.MROUJ_summary(self.MROUJ(theta))
            
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
    
    def MROUJ(self, theta):
        obtime  = np.arange(0,self.n+1)/self.n * self.n * self.delta
        kappa, beta, sigma, lamb, mu = theta[:,0], theta[:,1], theta[:,2], theta[:,3], theta[:,4]
        m = 50
        L_tmp = kappa.size(0)
        y0 = torch.zeros(L_tmp)
        z0 = y0
        path = torch.zeros(L_tmp, obtime.size)
        path[:,0] = z0

        for l in range(len(obtime)-1):
            # X, Y generating
            del_x = obtime[l+1] - obtime[l]
            del_y = del_x / m

            for j in range(m):
                ran_num = torch.normal(0 * torch.ones(L_tmp), torch.ones(L_tmp))
                ran_num2 = Exponential(mu * torch.ones(L_tmp)).sample() # rate
                ran_num3 = torch.poisson(torch.ones(L_tmp) * lamb * del_y)
                z0 = z0 + kappa*(beta-z0)*del_y + sigma * ran_num * del_y ** (1/2) + ran_num2 * ran_num3
            path[:,l+1] = z0
        return(path)    

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

    def MROUJ_summary(self, X):
        """
        X: torch size: [L,n]
        """
        L0 = X.size()[0]
        n0 = X.size()[1]
        
        Xi = X[:,range(1,n0)]
        Xi1 = X[:,range(0,n0-1)]
        
        s0 = torch.mean(Xi, 1)
        s1 = torch.mean(Xi1, 1)
        
        Xi = Xi - torch.reshape(s0, (L0, 1))
        Xi1 = Xi1 - torch.reshape(s1, (L0, 1))
        
        s2 = torch.mean(Xi * Xi1, 1) / n0
        s3 = torch.mean(Xi **2 , 1) /n0
        s4 = torch.mean(Xi1 **2 , 1) /n0
        
        s5 = torch.mean(torch.abs(Xi - Xi1), 1)
        s6 = torch.mean((Xi - Xi1)**2 , 1) / n0 
        s7 = torch.mean((Xi - Xi1)**3 , 1) / n0
        s8 = torch.mean((Xi - Xi1)**4 , 1) / n0 ** 2
        
        s9 = torch.mean((Xi - Xi1)**2 * Xi, 1) / n0 
        s10 = torch.mean((Xi - Xi1)**2 * Xi ** 2, 1)/ n0 ** 2
        
        Xi = Xi + torch.reshape(s0, (L0, 1))
        Xi1 = Xi1 + torch.reshape(s1, (L0, 1))
        
        s11 = torch.mean(Xi * Xi1, 1) - s0 * s1
        s11 = s11 / ( torch.mean(Xi1 ** 2, 1) - s1 ** 2 )
        
        s12 = s0 * torch.mean(Xi1 ** 2,1) - s1 * torch.mean(Xi * Xi1, 1)
        s12 = s12 / (torch.mean(Xi1 ** 2, 1) - s1 ** 2)
        
        # Jump intensity
        tmp = abs(Xi - Xi1)
        
        thres = [1e-5 * 3, 1e-5 * 6, 1e-5 * 9, 1e-4 * 3, 1e-4 * 6, 1e-4 * 9, 1e-3 * 3, 1e-3 * 6, 1e-3 * 9,
            1e-2 * 3, 1e-2 * 6, 1e-2 * 9, 1e-1 * 3, 1e-1 * 6, 1e-1 * 9,
                1.0, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25]
        thres_tmp = []
        for i in range(len(thres)):
            temp = torch.sum( (tmp > thres[i] ), 1) /n0
            thres_tmp.append(temp)

        j_int = torch.column_stack(thres_tmp)
        
        # Jump magnitude
        tmp = Xi - Xi1
        num = 33
        q = []
        for i in range(num+1):
            q.append(i/num)
        
        q = torch.tensor(q)
        mag_q = torch.transpose(torch.quantile(tmp, q, 1), 0, 1)
        
        return(torch.column_stack((s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, 
                                s10, s11, s12, j_int, mag_q)) ) 
