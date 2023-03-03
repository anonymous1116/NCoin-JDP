############################################################
#### CIR stationary sialation Sialation with MLE,GMM  ####
############################################################

###############################################
#### CIR simulation with GMM  ####
###############################################

# Source ------------------------------------------------------------------
source("./Simulation/DESP_functions.R")

n_col = c(200, 500, 2000)
param_mat = rbind(c(3,2,1), c(4,1.5,1.5), c(2,1.1,1.9))

for (scenario in c(1,2,3)){
  for (n in n_col){
    sim_name = paste0("./Simulation/OU/simdata/sim_data_", n, "_", scenario,".csv")
    sim_data = read.csv(sim_name)[2:(n+2)]
    colnames(sim_data)<-NULL
    sim_data = as.matrix(sim_data)
    sim_num = dim(sim_data)[1]
    
    set.seed(123)
    
    param = param_mat[scenario,] # a, b, sigma
    mu_sim<-c()
    theta_sim<-c()
    sigma_sim<-c()
    
    delta = 1/12
    
    for (sim in 1:sim_num){
      Y=sim_data[sim,]
      results = OU_estimation(Y, delta = delta)
      mu_sim[sim] = max(min(results$mu,5),1)
      theta_sim[sim] = results$theta
      sigma_sim[sim] = results$sigma
    }
    cat("Scenario", scenario, "n=", n, ", param=", param,"\n",
        c("a:    ", round(mean(mu_sim), 4),    "bias :", round(mean(mu_sim-param[1]), 4), 
          "r.bias :", round(mean(mu_sim-param[1])/param[1],4), 
          "sd:", round(sd(mu_sim),4),"\n",
          "b: ", round(mean(theta_sim), 4), "bias :", round(mean(theta_sim - param[2]), 4), 
          "r.bias :", round(mean(theta_sim-param[2])/param[2],4),
          "sd:",round(sd(theta_sim),4), "\n",
          "sigma: ", round(mean(sigma_sim), 4), "bias :", round(mean(sigma_sim -param[3]), 4), 
          "r.bias :", round(mean(sigma_sim-param[3])/param[3],4),
          "sd:",round(sd(sigma_sim),4),"\n" ) )
    cat("RMSE: ", sqrt((sum((mu_sim-param[1])^2) +  sum((theta_sim-param[2])^2) + sum((sigma_sim-param[3])^2))/sim_num), "\n")
    
  }
}


