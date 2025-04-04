import torch
import os
import math
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
                "CIR_summary": {"x0_list": x0_list if x0_list else [],  
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
        elif self.task == 'CIR':
            return self.CIR()
        elif self.task == 'OU_summary':
            return self.OU()
        elif self.task == 'CIR_summary':
            return self.CIR()
        elif self.task == 'MROUJ':
            return self.MROUJ()
        elif self.task == 'MROUJ_summary':
            return self.MROUJ()
        
    def OU(self):
        return BoxUniform(low=torch.tensor([1, 1, 0.5]), high=torch.tensor([5, 2.5, 2]))
    
    def CIR(self):
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
        elif self.task == "CIR":
            return self.CIR(theta)
        elif self.task == "PBJD":
            return self.PBJD(theta)
        elif self.task =="MROUJ":
            return self.MROUJ(theta)

        elif self.task =="OU_summary":
            return self.OU_summary(self.OU(theta))
        elif self.task =="CIR_summary":
            return self.CIR_summary(self.CIR(theta))
        elif self.task =="PBJD_summary":
            return self.PBJD_summary(self.PBJD(theta))
        elif self.task =="MROUJ_summary":
            return self.MROUJ_summary(self.MROUJ(theta))
        
    def OU(self, theta, batch_size=1_000_000):
        n = self.n
        delta = self.delta
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        L_OU = theta.size(0)
        time_OU = torch.linspace(0, n * delta, n + 1)  # (n+1) time steps

        mu_OU, theta_OU, sigma2_OU = theta[:, 0], theta[:, 1], theta[:, 2]

        # Initialize an empty list to store CPU results
        path_OU_list = []

        # Process in batches to avoid memory overload
        for start in range(0, L_OU, batch_size):
            end = min(start + batch_size, L_OU)
            
            # Process batch
            mu_batch = mu_OU[start:end].to(device)
            theta_batch = theta_OU[start:end].to(device)
            sigma2_batch = sigma2_OU[start:end].to(device)

            # Compute standard deviation for initial state
            std_init = torch.sqrt(sigma2_batch / (2 * mu_batch))

            # Initialize batch paths (Allocate **directly on CPU**)
            path_batch = torch.empty((end - start, n + 1), dtype=torch.float32, device="cpu")

            # Initialize first value of the path
            z0 = torch.normal(theta_batch, std_init)
            path_batch[:, 0] = z0.cpu()  # Store on CPU
            
            del std_init  # Free GPU memory
            torch.cuda.empty_cache()

            # Compute time step difference once
            del_L = time_OU[1] - time_OU[0]
            exp_neg_mu_del = torch.exp(-mu_batch * del_L)
            sqrt_term = torch.sqrt(sigma2_batch / (2 * mu_batch) * (1 - exp_neg_mu_del**2))
            # Compute the rest of the path
            for l in range(1, n + 1):
                OU_mean = z0 * exp_neg_mu_del + theta_batch * (1 - exp_neg_mu_del)
                z0 = torch.normal(OU_mean, sqrt_term)  # Update recursively
                
                # Store result **directly** in preallocated CPU tensor
                path_batch[:, l] = z0.cpu()

            # Store batch results
            path_OU_list.append(path_batch)

            # Free GPU memory
            del mu_batch, theta_batch, sigma2_batch, exp_neg_mu_del, sqrt_term, z0, path_batch
            torch.cuda.empty_cache()
    
            
        # Concatenate all batches on CPU
        return torch.row_stack(path_OU_list)
        
    def CIR(self, theta):
        L_CIR = theta.size(0)
        time_CIR = np.arange(0,self.n+1)/self.n * self.n * self.delta

        a, b, sigma2 = theta[:,0], theta[:, 1], theta[:, 2]
        z0 = torch.ones(L_CIR)
        path_OU = torch.zeros(L_CIR, time_CIR.size)
        path_OU[:,0] = z0
        
        path = torch.zeros(L_CIR, time_CIR.size)
        path[:,0] = z0
        
        nu0 = 4 * a * b / sigma2
        nu0 = nu0.numpy()
        
        for l in range(time_CIR.size-1):
            del_L = time_CIR[l+1] - time_CIR[l]
            c0 = 4 * a / sigma2 / (1- torch.exp(-a * del_L))
            lambda0 = c0 * z0 * torch.exp(-a * del_L)
            lambda0 = lambda0.numpy()
            tmp = np.random.noncentral_chisquare(nu0, lambda0)
            tmp = torch.from_numpy(tmp)
            z0 = tmp/c0
            path[:,l+1] = z0
        return(path)

    def PBJD(self, theta, batch_size=500000):
        """
        Efficient PBJD simulator for large L by splitting into mini-batches.

        Args:
            theta (Tensor): shape (L, 6)
            batch_size (int): number of samples per batch (default: 10000)

        Returns:
            path (Tensor): shape (L, T) where T = self.n + 1
        """
        L_total = theta.shape[0]
        all_paths = []

        for start in range(0, L_total, batch_size):
            end = min(start + batch_size, L_total)
            theta_batch = theta[start:end]

            # Simulate one batch and append
            path_batch = self.PBJD_simulate_batch(theta_batch)
            all_paths.append(path_batch)

        # Concatenate all batches
        return torch.cat(all_paths, dim=0)
    
    def PBJD_simulate_batch(self, theta):
        """
        This function generates a one sample path between an interval for 
        process
        dX_t = muX_tdt + sigma dB_t + J_1t + J_2t 
        m : num of slice of each interval
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        torch.set_default_device(device)
        theta = theta.to(device)
        n = self.n
        delta = self.delta
        beta, sigma, lamb_p, lamb_n, eta_p, eta_n = theta[:,0], theta[:,1], theta[:,2], theta[:,3], theta[:,4], theta[:,5]
        obtime = np.arange(0,n+1)/n * n * delta

        L_tmp = theta.size()[0]
        z0 = torch.zeros(L_tmp)
        path = torch.zeros(L_tmp, obtime.size)
        
        for l in range(len(obtime)-1):
            # X, Y generating
            del_x = obtime[l+1] - obtime[l]

            ran_num = torch.normal(0 * torch.ones(L_tmp), torch.ones(L_tmp))
                
            # jump to positive
            N = torch.poisson(lamb_p * del_x)
            
            gamma = torch.distributions.Gamma(N.clamp(min=1), eta_p)
            J = gamma.sample()

            # Set J = 0 where N == 0
            J[N == 0] = 0.0
            
            J = J.clamp(max=1000)
            
            del N
            
            # jump to negative
            N = torch.poisson(lamb_n * del_x)
            gamma = torch.distributions.Gamma(N.clamp(min=1), eta_n)
            J2 = gamma.sample()
            
            # Set J = 0 where N == 0
            J2[N == 0] = 0.0
            
            J2 = J2.clamp(max = 1000)
            
            z0 = z0 + (beta - sigma ** 2/ 2) * del_x + sigma * ran_num * del_x ** (1/2) + J - J2
            
            path[:,l+1] = z0
        torch.set_default_device("cpu")
        return(path.to("cpu"))


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

    def OU_summary(self, X, batch_size=10000, device="cuda"):
        """
        Compute OU summary statistics in batches to avoid OOM.

        Args:
            X (Tensor): shape (L, n), potentially large
            batch_size (int): number of rows to process at a time
            device (str): 'cuda' or 'cpu' (if CUDA memory is too limited)

        Returns:
            Tensor: shape (L, 5)
        """
        L = X.size(0)
        n = X.size(1)
        summaries = []

        for start in range(0, L, batch_size):
            end = min(start + batch_size, L)
            X_batch = X[start:end].to(device)

            # Vectorized computations
            X_prev = X_batch[:, :-1]
            X_next = X_batch[:, 1:]

            sum1 = torch.sum(X_next * X_prev, dim=1)
            sum2 = torch.sum(X_next, dim=1)
            sum3 = torch.sum(X_prev, dim=1)
            sum4 = torch.sum(X_prev ** 2, dim=1)
            sum5 = torch.sum(X_next ** 2, dim=1)

            S1 = (sum1 - sum2 * sum3 / n) / n
            S2 = sum2 / n
            S3 = sum3 / n
            S4 = sum4 / n - (sum3 / n) ** 2
            S5 = sum5 / n - (sum2 / n) ** 2

            summary_batch = torch.stack((S1, S2, S3, S4, S5), dim=1).to("cpu")
            summaries.append(summary_batch)

            del X_batch, X_prev, X_next, S1, S2, S3, S4, S5, summary_batch
            torch.cuda.empty_cache()  # optional: help reduce GPU fragmentation

        return torch.cat(summaries, dim=0)    

    def CIR_summary(self, X):
        """
        X: torch size: [L,n]
        """
        
        X0 = X[:,:-1]
        X1 = X[:,1:]
        
        s0 = torch.mean(X0, 1, keepdim=True) # mean of x_{i-1}
        s1 = torch.mean(X1, 1, keepdim=True) # mean of x_i
        
        s2 = torch.mean((X0 - s0) * (X1 - s1),1,keepdim=True)
        s3 = torch.mean((X0 - s0)**2,1, keepdim=True)
        s4 = torch.mean((X1 - s1)**2,1, keepdim=True)
        
        s5 = torch.mean(1/ X0, 1, keepdim=True)
        s6 = torch.mean(X1/ X0, 1, keepdim=True)
        s7 = torch.mean(X1 ** 2/ X0, 1, keepdim=True)
        s8 = torch.log(torch.max(X0, 1e-10*torch.ones(1)))
        s8 = torch.mean(s8, 1, keepdim=True)
        
        s9 = torch.mean((X0 - s0)**2 * (X1 - s1), 1, keepdim=True)
        s10 = torch.mean((X0 - s0) * (X1 - s1)**2, 1, keepdim=True)
        s11 = torch.mean((X0 - s0)**2 * (X1 - s1)**2, 1, keepdim=True)
        return(torch.column_stack((s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11)) ) 

    def PBJD_summary(self, X):
        """
        X: torch size: [L,n]
        """
        delta = self.delta
        L0 = X.size()[0]
        n0 = X.size()[1]
        
        Xi = X[:,range(1,n0)]
        Xi1 = X[:,range(0,n0-1)]
        
        # mean
        s0 = torch.sum((Xi - Xi1), 1) / (delta* (n0-1))
        s1 = torch.mean(torch.abs(Xi - Xi1), 1) / n0
        s2 = torch.mean((Xi - Xi1)**2 , 1) / n0
        s3 = torch.mean((Xi - Xi1)**3 , 1) / n0 
        s4 = torch.mean((Xi - Xi1)**4 , 1) / n0 ** 2
        
        tmp = (Xi - Xi1 - torch.reshape(s0, (L0, 1)) * delta) ** 2/delta
        
        # sigma
        s5 = torch.mean(tmp, 1)/n0
        
        # Jump intensity
        tmp = (Xi - Xi1)
        
        thres = [1e-7 * 3, 1e-7 * 6, 1e-7 * 9, 
                1e-6 * 3, 1e-6 * 6, 1e-6 * 9, 
                1e-5 * 3, 1e-5 * 6, 1e-5 * 9, 
                1e-4 * 3, 1e-4 * 6, 1e-4 * 9, 
                1e-3 * 3, 1e-3 * 6, 1e-3 * 9,
                1e-2 * 3, 1e-2 * 6, 1e-2 * 9]
        thres_tmp = []
        for i in range(len(thres)):
            temp = torch.sum( (tmp > thres[i] ), 1) /n0
            thres_tmp.append(temp)
        
        j_int1 = torch.column_stack(thres_tmp)
        
        thres_tmp2 = []
        for i in range(len(thres)):
            temp = torch.sum( (tmp < -thres[i] ), 1) /n0
            thres_tmp2.append(temp)
        
        j_int2 = torch.column_stack(thres_tmp2)
        
        # Jump magnitude
        tmp = Xi - Xi1
        num = 33
        q = []
        for i in range(num+1):
            q.append(i/num)
        
        q = torch.tensor(q)
        mag_q = torch.transpose(torch.quantile(tmp, q, 1), 0, 1)
        
        return(torch.column_stack((s0, s1, s2, s3, s4, s5, 
                                j_int1, j_int2, mag_q)) ) 


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


def truncated_normal(shape, mean=0.0, std=1.0, lower=-0.5, upper=0.5):
    """
    Generates samples from a truncated normal distribution in O(1) time using inverse CDF method.
    
    Returns:
    - Tensor of shape `shape` with samples from the truncated normal distribution.
    """
    # Convert lower and upper bounds to standard normal space
    lower_cdf = 0.5 * (1 + math.erf((lower - mean) / (std * math.sqrt(2))))
    upper_cdf = 0.5 * (1 + math.erf((upper - mean) / (std * math.sqrt(2))))

    # Sample uniformly in the truncated CDF range
    uniform_samples = torch.rand(shape, dtype=torch.float32) * (upper_cdf - lower_cdf) + lower_cdf

    # Apply inverse CDF (probit function) using erfinv
    truncated_samples = mean + std * torch.erfinv(2 * uniform_samples - 1) * math.sqrt(2)

    return truncated_samples

def truncated_exponential(shape, rate=1.0, lower=0.0, upper=1.0):
    """
    Generates samples from a truncated exponential distribution using the inverse CDF method.

    Parameters:
    - shape (tuple): Output shape of the tensor.
    - rate (float): Rate parameter (λ) of the exponential distribution.
    - lower (float): Lower truncation bound (≥ 0).
    - upper (float): Upper truncation bound (> lower).

    Returns:
    - Tensor of shape `shape` with samples from the truncated exponential distribution.
    """
    if lower < 0 or upper <= lower:
        raise ValueError("Invalid truncation bounds. Ensure 0 <= lower < upper.")

    # Compute CDF values at truncation bounds
    lower_cdf = 1 - math.exp(-rate * lower)
    upper_cdf = 1 - math.exp(-rate * upper)

    # Sample uniformly between CDF bounds
    uniform_samples = torch.rand(shape, dtype=torch.float32) * (upper_cdf - lower_cdf) + lower_cdf

    # Apply inverse CDF (quantile function) of exponential distribution
    truncated_samples = -torch.log(1 - uniform_samples) / rate

    return truncated_samples.clamp(min=1e-15)


def PBJD_truncated_priors(L, param, trunc):
    
    def fallback(trunc_val, default_val):
        return trunc_val if trunc_val is not None else default_val

    # Unpack parameters and truncation ranges
    beta_range, sigma_param, lamb_p_param, lamb_n_param, eta_p_param, eta_n_param = param
    b_range, s_range, lp_range, ln_range, ep_range, en_range = trunc

    # Apply defaults if None
    b_range  = fallback(b_range, beta_range)
    s_range  = fallback(s_range, [0.0, float('inf')])
    lp_range = fallback(lp_range, [0.0, float('inf')])
    ln_range = fallback(ln_range, [0.0, float('inf')])
    ep_range = fallback(ep_range, [0.0, float('inf')])
    en_range = fallback(en_range, [0.0, float('inf')])

    # Sample from priors
    b_ran  = torch.rand(L) * (b_range[1] - b_range[0]) + b_range[0]
    s_ran  = truncated_exponential((L,), rate=sigma_param[0],   lower=s_range[0],  upper=s_range[1])
    lp_ran = truncated_exponential((L,), rate=lamb_p_param[0], lower=lp_range[0], upper=lp_range[1])
    ln_ran = truncated_exponential((L,), rate=lamb_n_param[0], lower=ln_range[0], upper=ln_range[1])
    ep_ran = truncated_exponential((L,), rate=eta_p_param[0],  lower=ep_range[0], upper=ep_range[1])
    en_ran = truncated_exponential((L,), rate=eta_n_param[0],  lower=en_range[0], upper=en_range[1])

    # Stack and transform
    theta_transform = torch.stack((
        b_ran,
        s_ran,
        lp_ran,
        ln_ran,
        ep_ran,
        en_ran
    ), dim=1)

    return theta_transform

def PBJD_truncated_priors2(L, param, trunc):
    
    def fallback(trunc_val, default_val):
        return trunc_val if trunc_val is not None else default_val

    # Unpack parameters and truncation ranges
    beta_range, sigma_param, lambda_p_range, lambda_n_range, eta_p_param, eta_n_param = param
    b_range, s_range, lp_range, ln_range, ep_range, en_range = trunc

    # Apply defaults if None
    b_range  = fallback(b_range, beta_range)
    s_range  = fallback(s_range, [0.0, float('inf')])
    lp_range  = fallback(lp_range, lambda_p_range)
    ln_range  = fallback(ln_range, lambda_n_range)
    ep_range = fallback(ep_range, [0.0, float('inf')])
    en_range = fallback(en_range, [0.0, float('inf')])

    # Sample from priors
    b_ran  = torch.rand(L) * (b_range[1] - b_range[0]) + b_range[0]
    s_ran  = truncated_exponential((L,), rate=sigma_param[0],   lower=s_range[0],  upper=s_range[1])
    lp_ran =  torch.rand(L) * (lp_range[1] - lp_range[0]) + lp_range[0]
    ln_ran =  torch.rand(L) * (ln_range[1] - ln_range[0]) + ln_range[0]
    ep_ran = truncated_exponential((L,), rate=eta_p_param[0],  lower=ep_range[0], upper=ep_range[1])
    en_ran = truncated_exponential((L,), rate=eta_n_param[0],  lower=en_range[0], upper=en_range[1])

    # Stack and transform
    theta_transform = torch.stack((
        b_ran,
        s_ran,
        lp_ran,
        ln_ran,
        ep_ran,
        en_ran
    ), dim=1)

    return theta_transform


def PBJD_theta_log_transform(tmp):
    transformed = torch.cat([
        tmp[:, [0]],             # Keep column 0 (shape: (L, 1))
        torch.log(tmp[:, 1:])    # Exponentiate columns 1 to 5 (shape: (L, 5))
    ], dim=1)
    return transformed

def PBJD_theta_exp_transform(tmp):
    transformed = torch.cat([
        tmp[:, [0]],             # Keep column 0 (shape: (L, 1))
        torch.exp(tmp[:, 1:])    # Exponentiate columns 1 to 5 (shape: (L, 5))
    ], dim=1)
    return transformed
