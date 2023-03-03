############################################################
#### CIR stationary sialation Sialation with MLE,GMM  ####
############################################################

###############################################
#### CIR simulation with GMM  ####
###############################################

# Source ------------------------------------------------------------------
source("./Simulation/DESP_functions.R")

n_col = c(200, 500, 2000)
param_mat = rbind(c(3,2,0.5), c(4,1.5,0.8), c(2.5,2,0.2))

for (scenario in c(1,2,3)){
  for (n in n_col){
    sim_name = paste0("./Simulation/CIR/simdata/sim_data_CIR_", n, "_", scenario,".csv")
    sim_data = read.csv(sim_name)[2:(n+2)]
    colnames(sim_data)<-NULL
    sim_data = as.matrix(sim_data)
    sim_num = dim(sim_data)[1]
    
    set.seed(123)
    
    param = param_mat[scenario,] # a, b, sigma
    a_sim<-c()
    b_sim<-c()
    sigma_sim<-c()
    
    delta = 1/12
    
    for (sim in 1:sim_num){
      Y=sim_data[sim,]
      results = CIR_estimation(Y, delta = delta)
      a_sim[sim] = max(min(results$a,5),1)
      b_sim[sim] = results$b
      sigma_sim[sim] = results$sigma
    }
    cat("Scenario", scenario, "n=", n, ", param=", param,"\n",
        c("a:    ", round(mean(a_sim), 4),    "bias :", round(mean(a_sim-param[1]), 4), 
          "r.bias :", round(mean(a_sim-param[1])/param[1],4), 
          "sd:", round(sd(a_sim),4),"\n",
          "b: ", round(mean(b_sim), 4), "bias :", round(mean(b_sim - param[2]), 4), 
          "r.bias :", round(mean(b_sim-param[2])/param[2],4),
          "sd:",round(sd(b_sim),4), "\n",
          "sigma: ", round(mean(sigma_sim), 4), "bias :", round(mean(sigma_sim -param[3]), 4), 
          "r.bias :", round(mean(sigma_sim-param[3])/param[3],4),
          "sd:",round(sd(sigma_sim),4),"\n" ) )
    cat("RMSE: ", sqrt((sum((a_sim-param[1])^2) +  sum((b_sim-param[2])^2) + sum((sigma_sim-param[3])^2))/sim_num), "\n")
    
  }
}


