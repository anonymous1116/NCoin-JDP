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
Here, we consider four differnet processes, OUJ, SQRJ, PBJD, and BOUJ. First as before, we should generate synthetic data that is used for inference. To obtain the results in Figure 10, and Figure 11, we generate 7,500,000 synthetic data for each size $\in \{1000, 3000, 5000 \}$ and process. To obtain these, implement `JD_simulation generator.ipynb`. Since it takes numerous time to finish, we recommend you use some parallel computing for implementing these. 

And we get the NCoin-JDP estimator mapping using 250,000 synthetica data for each iteration (We iterate 10 times for each case.) Implement `MROUJ/MROUJ.ipynb`, `SQRJ/SQRJ.ipynb`, `PBJD/PBJD.ipynb` and `BOUJ/BOUJ.ipynb`. Before implementing, make folder to save learned networks. 

In order to improve reproducibility, here in Github direcotry, we make each folder and save learned nets. To obtain Figure 8, above two steps are not necessary. Implement `JDP_performance_eval2.ipynb`.

We implement Algorithm 2 to perform Bayesian inference for a sample path generated from each process. Implement `MROUJ/MROUJ_infer.ipynb`, `SQRJ/SQRJ_infer.ipynb`, `PBJD/PBJD_infer.ipynb` and `BOUJ/BOUJ_infer.ipynb`. For implementation, the generated synthetic data is essential. Recall `JD_simulation generator.ipynb` for the generation. 

For whom does not want to generate such files, we save our results. Go to `MROUJ/MROUJ_calibrate`, `SQRJ/SQRJ_calibrate`, `PBJD/PBJD_calibrate`, and `BOUJ/BOUJ_calibrate` for these.


### Section 5.4
In order to obtain Table 2 and Table 3, `JD_simulation generator2.ipynb` should be implemented first to secure synthetic data. For table 2, implement `Compare_MOUJ`. For table 3, implement `Compare_MOUJ_sigma_1`, `Compare_MOUJ_sigma_2`. 

The results are saved in `results` folder.

## Real Data Analysis (Section 6)
Using NCoin-JDP, we conduct a comparative study of jumps in daily closing prices of the NASDAQ and S\&P 500 between 1993 and 2024, focusing on the evaluation of economic parameters during the recent COVID-19 pandemic. We divide the periods into five segments, P1 to P5, each with a size of $n=2015$.
The first period, P1, spans 1992-01-12 to 2000-01-01, marking the recovery from the mild recession of the early 1990s. The second period, P2, is from 1998-01-01 to 2006-01-05, which includes the 9/11 attack that severely impacted the U.S. economy and triggered global market disruptions. This period also covers the dotcom bubble's rise and subsequent collapse between 2001 and 2002. The third period, P3, runs from 2005-01-01 to 2013-01-04, encompassing the 2007-2008 financial crisis that triggered the Great Recession and significantly affected both stock markets. The fourth period, P4, extends from 2011-01-01 to 2019-01-05, during which the US economy slowly recovered from the Great Recession and experienced steady growth. The final period, P5, is set from 2016-01-01 to 2024-01-05, including the Great Lockdown resulting from the COVID-19 pandemic in 2019. This division is chosen to delete seasonal effects, allowing us to analyze and compare the impact of major economic events on the stock market over the specified periods.

<img width="883" alt="image" src="https://github.com/anonymous1116/NCoin-JDP/assets/126707827/7eb74972-c7bb-4fd0-bb65-a76753a37f7d">

Implement `RDA_stocks.ipynb` to reproduce the results in Section 6. The results are uploaded in the folder `RDA_calibrate`.


