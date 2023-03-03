# NCoin-DP
## Tutorial
We first provide a tutorial for NCoin-DP. Refer to `NCoin-DP_tutorial.ipynb` above.

## Simulation
We provide codes for simulation and numerical results for NCoin-DP, MLE, Pseudo-MLE, and GMM. In our simulation studies, we consider four types of diffusion to evaluate the performance of NCoin-DP. These are  
- the OU process $dX_t = \kappa(\beta - X_t) dt + \sigma dB_t$
with an initial value $X_0 \sim N(\beta, \frac{\sigma^2}{2\kappa} )$ where $\kappa>0, \beta>0, \sigma>0$, 

- the CIR model $dX_t = \kappa (\beta - X_t) dt + \sigma \sqrt{X_t} dB_t$ where $X_0 = x_0$ and $2\kappa \beta \geq \sigma^2$ and $\kappa>0, \beta>0, \sigma>0$, 

- the Jacobi diffusion model $dX_t = \kappa (\beta - X_t) dt + \sigma \sqrt{X_t(1-X_t)} dB_t$ where $X_0 = x_0$ and $\kappa>0, 0<\beta<1, \sigma>0$, and 

- a 3-dimensional multivariate process $dX_t = -A(\theta)X_t + dW_t$ where $A(\theta)$ is a lower diagonal matrix.
