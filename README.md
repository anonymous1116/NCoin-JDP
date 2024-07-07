# Nerual Conformal Inference for Jump Diffusion Processes (NCoin-JDP)
## Tutorial
We first provide a tutorial for NCoin-DP. It contains

- How to generate synthetic data (Sec 2.1)
- Functions of input statistic (Sec 2.2)
- Learning the mapping using DNN (Sec 2.3)
- Learning checking procedure (Sec 2.4)
- Uncertainty characterization (Sec 3)

using OU process. For detailed, go to `OU_tutorial.ipynb` above.


## Simulation (Section 5)
### Section 5.1
In this section, we consider OU process to evaluate the performance of NCoin-DP. We provide codes for simulation and numerical results for NCoin-JDP, MLE, GMM, and MCMC. The NCoin-JDP, and MLE methods are implemented through `Python`, and the GMM and MCMC methods through `R`. `Jupyter Notebook`s are provided for `Python` code, and `R` scripts are also provided.

In order to reproduce the results in Section 5.1, first you should generate synthetic data and test data. Go to `simulaiton generator.ipynb` file and implement those. Make sure you generate folder `syn_data` and `test_data` for your appropriate directory, and change the code inside to direct the files to be saved in your assigned directory. Using the synthetic data, we implement the NCoin-JDP. The test data are evaluated by all of four methods.

For NCoin-JDP, implement `OU_sim.ipynb`. Before implemting this file, make `nets` folder to save the learned mappings. For GMM and MCMC, we implemented and uploaded the results in this directory, so just download the results. For MCMC, `MCMC_OU.txt` is uploaded. For GMM, go to `GMM/GMM_OU.txt`. The `R` source code is available upon request.

For two sensitivity analysis, implement `SensitivityAnalysis.ipynb`, and `SensitivityAnalysis2.ipynb`.

Finally, implement `Perform_eval.ipynb`, then you will get Figures in Section 5.1. 


### Section 5.2
For both P1 and P2, we generate synthetic data first. Implement `inference_generator.ipynb`, and then `infer_sim_0.pt` and `infer_sim_1.pt` will be generated, which are synthetic data for P1 and P2, respectively. 

Algorithm 1 is then implemented 20 times using `inference_learning.ipynb` and `inference_learning_1.ipynb` for P1 and P2, respectively. Make sure to make folder `infer_nets/net0` and `infer_nets/net1` to save the learned mappings for each.

Using these 20 mappings, we implement Algorithm 2 by implementing `OU_inference_sim.ipynb` and `OU_inference_sim_1.ipynb`. We vary hyperparameter $r = \{.05, .07, .10\}$. The results when $r = .05$ and $.07$ are uploaded, while the case of $r = .10$ cannot be uploaded due to the size issue in Github.

Table 1 can be obtained by implementing `Perform_eval_inference.ipynb`.


### Section 5.3
Here, we consider four differnet processes, OUJ, SQRJ, PBJD, and BOUJ. First as before, we should generate synthetic data that is used for inference. To obtain the results in Figure 8, we generate 2,500,000 synthetic data for each size $\in \{1000, 3000, 5000 \}$ and process. To obtain these, implement `JD_simulation generator.ipynb`. Since it takes numerous time to finish, we recommend you use some parallel computing for implementing these. 

And we get the NCoin-JDP estimator mapping using 250,000 synthetica data for each iteration (We iterate 10 times for each case.) Implement `MROUJ.ipynb`, `SQRJ.ipynb`, `PBJD.ipynb` and `BOUJ.ipynb`. Before implementing, make folder to save learned networks. Here we make each folder and learned nets to improve reproducibility for all cases. 


## Real Data Analysis (Section 6)
We collect adjusted closing price of MSFT and ADBE on n=1259 trading days between January 01, 2013 and December 31, 2017. See the image below. We model a pair trading schemes using those two data. For detailed analysis go to the folder, `RealDataAnalysis` and run `adbemscf.ipynb`


<img width="779" alt="Screenshot 2023-03-03 121408" src="https://user-images.githubusercontent.com/126707827/222784718-b72d35a1-33b0-44d3-bb47-769b1282e57f.png">
