###############################################
#### Jacobi diffusion simulation with GMM  ####
###############################################

# Source ------------------------------------------------------------------
source("./DESP_functions.R")

n_col = c(500, 1000, 3000)
param_mat = rbind(c(3,0.2,0.5), c(4,0.5,0.7), c(2,0.6,0.2))

for (scenario in c(1,2,3)){
  for (n in n_col){
    sim_name = paste0("./simdata/sim_data_Jacobi_", n, "_", scenario,".csv")
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
      init = param * runif(3,0.8,1.2)
      init = c(max(min(init[2], 0.7),0.2),max(min(init[1], 5),1), max(min(init[3], 0.7),0.1) )
      tmp<- GMM_Jacobi(Y, Delta = delta, par = init, maxiter =50)
      a_sim[sim]    = tmp$coefficients[2,1] 
      b_sim[sim] = tmp$coefficients[1,1]
      sigma_sim[sim] = tmp$coefficients[3,1]
      
      if (sim %% 100 == 0 ){
        cat("GMM sim num:", sim, "  ")
      }
    }
    cat("\n param: ", param_mat[scenario,], "n=", n, "\n",
        c("a: ", round(mean(a_sim), 4),    "bias :", round(mean(a_sim-param[1]), 4), 
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
