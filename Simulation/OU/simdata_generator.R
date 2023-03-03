# Simulation Data Generation

# Source ------------------------------------------------------------------
source("./Simulation/DESP_functions.R")

sim_num = 500

sim_n = c(200, 500, 2000)
delta_col<-c(1/12)
# Scenario 1
param = c(3,2,1) # mu, theta, sigma

# Scenario 2
param2 = c(4, 1.5, 1.5)

# Scenario 3
param3 = c(2, 1.1, 1.9)

set.seed(123)

# Generating simulation data used in this simulation
# and to transfer to the python code 
for (n in sim_n){
  sim_data <- c()
  for (sim in 1:sim_num){
    sim_data = rbind(sim_data,
                     OU_stnry_gnrtr(delta_col[1], n0 = n, mu = param[1], theta = param[2], sigma = param[3])
    )
  }
  sim_name = paste0("./Simulation/OU/simdata/sim_data_", n, "_1.csv")
  write.csv(sim_data, file = sim_name)
  
  sim_data <- c()
  for (sim in 1:sim_num){
    sim_data = rbind(sim_data,
                     OU_stnry_gnrtr(delta_col[1], n0 = n, mu = param2[1], theta = param2[2], sigma = param2[3])
    )
  }
  sim_name = paste0("./Simulation/OU/simdata/sim_data_", n, "_2.csv")
  write.csv(sim_data, file = sim_name)
  
  sim_data <- c()
  for (sim in 1:sim_num){
    sim_data = rbind(sim_data,
                     OU_stnry_gnrtr(delta_col[1], n0 = n, mu = param3[1], theta = param3[2], sigma = param3[3])
    )
  }
  sim_name = paste0("./Simulation/OU/simdata/sim_data_", n, "_3.csv")
  write.csv(sim_data, file = sim_name)
}
