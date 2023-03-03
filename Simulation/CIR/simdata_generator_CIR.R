# Simulation Data Generation

# Source ------------------------------------------------------------------
source("./Simulation/DESP_functions.R")

sim_num = 500

sim_n = c(200, 500, 2000)
delta_col<-c(1/12)
# Scenario 1
param = c(3,2,0.5) # mu, theta, sigma

# Scenario 2
param2 = c(4, 1.5, 0.8)

# Scenario 3
param3 = c(2.5, 2, 0.2)

set.seed(123)

# Generating simulation data used in this simulation
# and to transfer to the python code 
for (n in sim_n){
  sim_data <- c()
  for (sim in 1:sim_num){
    sim_data = rbind(sim_data,
                     CIR_nonstnry_gnrtr(delta_col[1], n0 = n, param[1], param[2],param[3], 0.1)
    )
  }
  sim_name = paste0("./Simulation/CIR/simdata/sim_data_CIR_", n, "_1.csv")
  write.csv(sim_data, file = sim_name)
  
  sim_data <- c()
  for (sim in 1:sim_num){
    sim_data = rbind(sim_data,
                     CIR_nonstnry_gnrtr(delta_col[1], n0 = n, param2[1], param2[2], param2[3],0.1)
    )
  }
  sim_name = paste0("./Simulation/CIR/simdata/sim_data_CIR_", n, "_2.csv")
  write.csv(sim_data, file = sim_name)
  
  sim_data <- c()
  for (sim in 1:sim_num){
    sim_data = rbind(sim_data,
                     CIR_nonstnry_gnrtr(delta_col[1], n0 = n, param3[1], param3[2], param3[3],0.1)
    )
  }
  sim_name = paste0("./Simulation/CIR/simdata/sim_data_CIR_", n, "_3.csv")
  write.csv(sim_data, file = sim_name)
}
