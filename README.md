# NCoin-DP
## Tutorial
We first provide a tutorial for NCoin-DP. Refer to `NCoin-DP_tutorial.ipynb` above.

## Simulation
We provide codes for simulation and numerical results for NCoin-DP, MLE, Pseudo-MLE, AOLS, and GMM. In our simulation studies, we consider four types of diffusion to evaluate the performance of NCoin-DP. These are  
- the OU process $dX_t = \kappa(\beta - X_t) dt + \sigma dB_t$
with an initial value $X_0 \sim N(\beta, \frac{\sigma^2}{2\kappa} )$ where $\kappa>0, \beta>0, \sigma>0$, 

- the CIR model $dX_t = \kappa (\beta - X_t) dt + \sigma \sqrt{X_t} dB_t$ where $X_0 = x_0$ and $2\kappa \beta \geq \sigma^2$ and $\kappa>0, \beta>0, \sigma>0$, 

- the Jacobi diffusion model $dX_t = \kappa (\beta - X_t) dt + \sigma \sqrt{X_t(1-X_t)} dB_t$ where $X_0 = x_0$ and $\kappa>0, 0<\beta<1, \sigma>0$, and 

- a 3-dimensional multivariate process $dX_t = -A(\theta)X_t + dW_t$ where $A(\theta)$ is a lower diagonal matrix.

Each folder is created for every process. Since we use both the `R` and `Python`, `R` for GMM and Pseudo-MLE, and `Python` for NCoin-DP, AOLS, and MLE, we create the 500 simulation data from `R` for each process, and save it as excel file. In order to reproduce the numerical results, generating simulation data may need to be preceded before actually implementing each estimator.

### To obtain the simulation results in Section 6, go to the folder `Simulation` above.
#### For OU process
go to `OU` and run
- NCoin-DP: `OU_n=200.ipynb`, `OU_n=500.ipynb`, `OU_n=2000.ipynb`
- MLE: `MLE_OU_simulation.R`
- GMM : `GMM_OU_simulation.R`

#### For CIR model
go to  `CIR` and run
- NCoin-DP A: `CIR-A_n=200.ipynb`, `CIR-A_n=500.ipynb`, `CIR-A_n=2000.ipynb`
- NCoin-DP B: `CIR-B_n=200.ipynb`, `CIR-B_n=500.ipynb`, `CIR-B_n=2000.ipynb`
- GMM : `GMM_CIR_simulation.R`
- Pseudo-MLE : `PseudoMLE_CIR_simulation.R`

#### For Jacobi diffusion model
go to  `Jacobi` and run
- NCoin-DP: `Jacobi_n=200.ipynb`, `Jacobi_n=500.ipynb`, `Jacobi_n=2000.ipynb`
- GMM : `GMM_Jacobi_simulation.R`

#### For the multivariate process


## Real Data Analysis
We collect adjusted closing price of MSFT $A_t$ and ADBE $C_t$ on $n=1259$ trading days between January 01, 2013 and December 31, 2017. We model the log-prices $S_t\equiv\log A_t$ and its log-difference $Y_t\equiv \log C_t - \log A_t$ as

$$dS_t = (\mu - \frac{1}{2}\sigma^2 )dt + \sigma dB^1_t  $$
$$dY_t = \kappa(\beta - Y_t )dt + \eta  dB^2_t      $$
with $E[dB^1_t dB^2_t] = \rho dt$.

