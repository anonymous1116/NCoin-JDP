OU_stnry_gnrtr<-function(del,n0, mu, theta, sigma){
  x0 = rnorm(1, mean = theta, sd = sqrt(sigma^2/mu))
  tmp = c()
  prex = x0
  for (i in 1:n0){
    mtx =  prex * exp(-mu*del) + theta * (1-exp(-mu*del))
    varx = sigma^2/(2* mu) * (1-exp(-2* mu * del))
    tmp[i] = rnorm(1, mean = mtx, sd = sqrt(varx))  
    prex = tmp[i]
  } 
  return(c(x0,tmp))
}


# non-stationary OU process generator, initial value = x0
OU_nonstnry_gnrtr<-function(del, n0, mu, theta, sigma, x0){
  tmp = c()
  prex = x0
  for (i in 1:n0){
    mtx =  prex * exp(-mu*del) + theta * (1-exp(-mu*del))
    varx = sigma^2/(2* mu) * (1-exp(-2* mu * del))
    tmp[i] = rnorm(1, mean = mtx, sd = sqrt(varx))  
    prex = tmp[i]
  } 
  return(c(x0,tmp))
}

# stationary CIR process generator, initial value = x0
CIR_stnry_gnrtr<-function(del, n0, a, b, sigma){
  x0 = rgamma(1, shape = 2*a*b / sigma^2, rate = 2 * a / sigma^2)
  tmp = c()
  prex = x0
  
  c0 = 4 * a / sigma^2 * 1/ (1- exp(-a * del))
  nu0 = 4 * a * b / sigma^2
  for (i in 1:n0){
    lambda0 = c0 * prex * exp(-a * del)
    tmp[i] = rchisq(1, df = nu0, ncp = lambda0) / c0  
    prex = tmp[i]
  } 
  return(c(x0,tmp))
}

# non-stationary CIR process generator, initial value = x0
CIR_nonstnry_gnrtr<-function(del, n0, a, b, sigma,x0){
  tmp = c()
  prex = x0
  c0 = 4 * a / sigma^2 * 1/ (1- exp(-a * del))
  nu0 = 4 * a * b / sigma^2
  for (i in 1:n0){
    lambda0 = c0 * prex * exp(-a * del)
    tmp[i] = rchisq(1, df = nu0, ncp = lambda0) / c0  
    prex = tmp[i]
  } 
  return(c(x0,tmp))
}


Jacobi_nonstnry_gnrtr<-function(del, n0, m0, a, b, sigma, x0){
  tmp = c()
  prex = x0
  ran_num = rnorm(n0* m0)
  del_tmp = del/m0
  for (i in 1:(n0*m0)){
    prex = prex + a*(b-prex)*del_tmp+sigma*(sqrt(prex*(1-prex)))*sqrt(del_tmp)*ran_num[i]+
      (sigma^2)/4*(1-2*prex) * (del_tmp *  (ran_num[i]^2 -1 ))# Milstein approximation
    prex = max(prex, 0.001)
    prex = min(prex, 0.999)
    tmp[i] = prex
  } 
  return(c(x0,tmp[(rep(1:n0)*m0)]))
}




# OU estimation
OU_estimation<-function(vec, delta){
  n_vec = length(vec)
  n0 = n_vec - 1
  vec_xi = vec[2:n_vec]
  vec_xim1 =  vec[1:(n_vec-1)]
  beta1 = (sum(vec_xi * vec_xim1) - 1/n0 * sum(vec_xi) * sum(vec_xim1))/
    (sum(vec_xim1^2) - 1/n0* (sum(vec_xim1))^2)
  beta2 = 1/n0 *( sum(vec_xi) - beta1 * sum(vec_xim1) )/
    (1-beta1)
  tmp = (vec_xi - beta1 * vec_xim1 - beta2 * (1-beta1))
  beta3 = 1/n0 * sum(  tmp^2   )
  if (beta1<10^(-10)){
    beta1 = 10^(-10)
  }
  mu = -1/delta * log(beta1)
  theta = beta2
  sigma = sqrt(2 * mu * beta3 / (1-beta1^2))
  
  return(list(mu = mu, theta = theta, sigma = sigma))
}

# Estimation of CIR process
CIR_estimation<-function(vec, delta){
  n_vec = length(vec)
  n0 = n_vec - 1
  vec_xi = vec[2:n_vec]
  vec_xim1 =  vec[1:(n_vec-1)]
  beta1 = (sum(vec_xi) *sum(1/vec_xim1) / n0^2 - sum(vec_xi * (1/vec_xim1) )/n0 )/ 
    (sum(vec_xim1) * sum(1/vec_xim1) /n0^2 - 1) 
  beta2 = (1/n0 * sum(vec_xi * 1/vec_xim1) - beta1) / ((1-beta1) * sum(1/vec_xim1) / n0 )
  tmp = (vec_xi - beta1 *  (vec_xim1) - beta2 * (1-beta1))^2
  beta3 = 1/n0 * sum(  tmp * (1/vec_xim1)    ) 
  
  if (beta1<10^(-10)){
    beta1 = 10^(-10)
  }
  a = -1/delta * log(beta1)
  b = beta2
  sigma = sqrt(2 * a * beta3 / (1-beta1^2) )
  
  return(list(a = a, b = b, sigma = sigma))
}


GMM_OU <- function(X, Delta = deltat(X), par = NULL, maxiter = 25) {
  
  tol1 <- 0.001
  tol2 <- 0.001
  
  ft <- function(x, y, Theta, Delta){
    c.mean <- Theta[1] + (y-Theta[1])*exp(-Theta[2]*Delta)
    c.var <- Theta[3]^2 * (1-exp(-2*Theta[2]*Delta))/(2*Theta[2])
    cbind(x-c.mean, y*(x-c.mean), c.var-(x-c.mean)^2, y*(c.var-(x-c.mean)^2))
  }
  
  if (is.null(par)) {
    y    <- diff(X) / Delta
    init <- summary(lm(y ~ X[-length(X)]))
    par  <- c(init$coefficients[1,1], -init$coefficients[2,1], init$sigma * sqrt(Delta))
  }
  
  n  <- length(X)
  gn <- function(theta) apply(ft(X[2:n], X[1:(n - 1)], theta, Delta), 2, mean)
  Q  <- function(theta) sum(gn(theta)^2)
  S  <- function(j, theta) ((t(ft(X[(j + 2):n], X[(j + 1):(n - 1)], theta, Delta)) %*% ft(X[2:(n - j)], X[1:(n - j - 1)], theta, Delta))/n)
  Q1 <- function(theta, W) gn(theta) %*% W %*% gn(theta)
  
  ell <- n - 2
  w   <- 1 - (1:ell)/(ell + 1)
  
  theta0 <- optim(par, Q, method = "L-BFGS-B", lower = c(1, 1, 0.5), upper = c(2.5, 5, 2) )$par
  
  go   <- TRUE
  iter <- 0
  while (go) {
    iter  <- iter + 1
    S.hat <- S(0, theta0)
    for (i in 1:ell) S.hat = S.hat + w[i] * (S(i, theta0) + t(S(i, theta0)))
    W      <- solve(S.hat)
    est    <- optim(theta0, Q1, W = W, method = "L-BFGS-B", lower = c(1, 1, 0.5), upper = c(2.5, 5, 2))
    theta1 <- est$par
    if (sum(abs(theta0 - theta1)) < tol1 || est$value < tol2 || iter > maxiter) go <- FALSE
    theta0 <- theta1
  }
  
  dhat <- numDeriv::jacobian(gn, theta0)
  se   <- sqrt(diag(solve(t(dhat) %*% W %*% dhat)) / n)
  
  coeff <- cbind(theta0, se)
  rownames(coeff) <- c("alpha", "kappa", "sigma")
  colnames(coeff) <- c("Estimate", "Std. Error")
  res <- list(coefficients = coeff)
  class(res) <- "estVAS"
  return(res)
}


GMM_CIR<- function(X, Delta = deltat(X), par = NULL, maxiter = 25) {
  
  tol1 <- 0.001
  tol2 <- 0.001
  
  ft <- function(x, y, Theta, Delta){
    c.mean <- Theta[1] + (y-Theta[1])*exp(-Theta[2]*Delta)
    c.var <- Theta[3]^2 / (2*Theta[2]) * (1- exp(- 2* Theta[2]* Delta) ) * y 
    cbind(x-c.mean, y*(x-c.mean), c.var-(x-c.mean)^2, y*(c.var-(x-c.mean)^2))
  }
  
  if (is.null(par)) {
    y    <- diff(X) / Delta
    init <- summary(lm(y ~ X[-length(X)]))
    par  <- c(init$coefficients[1,1], -init$coefficients[2,1], init$sigma * sqrt(Delta))
  }
  
  n  <- length(X)
  gn <- function(theta) apply(ft(X[2:n], X[1:(n - 1)], theta, Delta), 2, mean)
  Q  <- function(theta) sum(gn(theta)^2)
  S  <- function(j, theta) ((t(ft(X[(j + 2):n], X[(j + 1):(n - 1)], theta, Delta)) %*% ft(X[2:(n - j)], X[1:(n - j - 1)], theta, Delta))/n)
  Q1 <- function(theta, W) gn(theta) %*% W %*% gn(theta)
  
  ell <- n - 2
  w   <- 1 - (1:ell)/(ell + 1)
  
  theta0 <- optim(par, Q, method = "L-BFGS-B", lower = c(1, 2, 0.1), upper = c(2.5, 5, 1) )$par
  
  go   <- TRUE
  iter <- 0
  while (go) {
    iter  <- iter + 1
    S.hat <- S(0, theta0)
    for (i in 1:ell) S.hat = S.hat + w[i] * (S(i, theta0) + t(S(i, theta0)))
    W      <- solve(S.hat)
    est    <- optim(theta0, Q1, W = W, method = "L-BFGS-B", lower = c(1, 2, 0.1), upper = c(2.5, 5, 1))
    theta1 <- est$par
    if (sum(abs(theta0 - theta1)) < tol1 || est$value < tol2 || iter > maxiter) go <- FALSE
    theta0 <- theta1
  }
  
  dhat <- numDeriv::jacobian(gn, theta0)
  se   <- sqrt(diag(solve(t(dhat) %*% W %*% dhat)) / n)
  
  coeff <- cbind(theta0, se)
  rownames(coeff) <- c("alpha", "kappa", "sigma")
  colnames(coeff) <- c("Estimate", "Std. Error")
  res <- list(coefficients = coeff)
  class(res) <- "estCIR"
  return(res)
}

GMM_Jacobi <- function(X, Delta = deltat(X), par = NULL, maxiter = 25) {
  
  tol1 <- 0.001
  tol2 <- 0.001
  
  #ft <- function(x, y, Theta, Delta){
  #  k1 <- Theta[1]
  #  k2 <- Theta[1]*(2* Theta[2] * Theta[1] + Theta[3]^2)/(2* Theta[2]+ Theta[3]^2)
  #  k3 <- Theta[1]*(2* Theta[2] * Theta[1] + Theta[3]^2)/(2* Theta[2]+ Theta[3]^2)*
  #    (2* Theta[2] * Theta[1] + 2*Theta[3]^2)/(2* Theta[2]+ 2*Theta[3]^2)
  #  k4 <- Theta[1]*(2* Theta[2] * Theta[1] + Theta[3]^2)/(2* Theta[2]+ Theta[3]^2)*
  #    (2* Theta[2] * Theta[1] + 2*Theta[3]^2)/(2* Theta[2]+ 2*Theta[3]^2)*
  #    (2* Theta[2] * Theta[1] + 3*Theta[3]^2)/(2* Theta[2]+ 3*Theta[3]^2)
  #  k11 <- exp(-Theta[2] * Delta) * (Theta[1]*(2* Theta[2] * Theta[1] + Theta[3]^2)/(2* Theta[2]+ Theta[3]^2)) + (1-exp(-Theta[2] * Delta) * Theta[1] )
  #  k12 <- exp(-Theta[2] * Delta) * (Theta[1]*(2* Theta[2] * Theta[1] + Theta[3]^2)/(2* Theta[2]+ Theta[3]^2)*
  #                                     (2* Theta[2] * Theta[1] + 2*Theta[3]^2)/(2* Theta[2]+ 2*Theta[3]^2)
  #  ) + (1-exp(-Theta[2] * Delta) * Theta[1] * (Theta[1]*(2* Theta[2] * Theta[1] + Theta[3]^2)/(2* Theta[2]+ Theta[3]^2)) )
  #  
  #  cbind(x-k1, x^2-k2, x^3-k3, x^4-k4, x*y - k11, x*y*y - k12)
  #}
  ft <- function(x, y, Theta, Delta){
    c.mean <- Theta[1] + (y-Theta[1])*exp(-Theta[2]*Delta)
    c.var <- Theta[3]^2 / (2*Theta[2]) * (1- exp( - 2* Theta[2]* Delta )) * y * (1-y)
    cbind(x-c.mean, y*(x-c.mean), c.var-(x-c.mean)^2, y*(c.var-(x-c.mean)^2))
  }
  
  #if (is.null(par)) {
  #  y    <- diff(X) / Delta
  #  init <- summary(lm(y ~ X[-length(X)]))
  #  par  <- c(init$coefficients[1,1], -init$coefficients[2,1], init$sigma * sqrt(Delta))
  #}
  
  n  <- length(X)
  gn <- function(theta) apply(ft(X[2:n], X[1:(n - 1)], theta, Delta), 2, mean)
  Q  <- function(theta) sum(gn(theta)^2)
  S  <- function(j, theta) ((t(ft(X[(j + 2):n], X[(j + 1):(n - 1)], theta, Delta)) %*% ft(X[2:(n - j)], X[1:(n - j - 1)], theta, Delta))/n)
  Q1 <- function(theta, W) gn(theta) %*% W %*% gn(theta)
  
  ell <- n - 2
  w   <- 1 - (1:ell)/(ell + 1)
  
  theta0 <- optim(par, Q, method = "L-BFGS-B", lower = c(0.2,1,0.1), upper = c(0.7,5,0.7))$par
  
  go   <- TRUE
  iter <- 0
  while (go) {
    iter  <- iter + 1
    S.hat <- S(0, theta0)
    for (i in 1:ell) S.hat = S.hat + w[i] * (S(i, theta0) + t(S(i, theta0)))
    W      <- solve(S.hat)
    est    <- optim(theta0, Q1, W = W, method =  "L-BFGS-B", lower = c(0.2,1,0.1), upper = c(0.7,5,0.7))
    theta1 <- est$par
    if (sum(abs(theta0 - theta1)) < tol1 || est$value < tol2 || iter > maxiter) go <- FALSE
    theta0 <- theta1
  }
  
  dhat <- numDeriv::jacobian(gn, theta0)
  se   <- sqrt(diag(solve(t(dhat) %*% W %*% dhat)) / n)
  
  coeff <- cbind(theta0, se)
  rownames(coeff) <- c("alpha", "kappa", "sigma")
  colnames(coeff) <- c("Estimate", "Std. Error")
  res <- list(coefficients = coeff)
  class(res) <- "estJacobi"
  return(res)
}

Multi_nonstnry_gnrtr<-function(del, n0, m0, mu0, sigma0, kappa0, theta0, eta0, rho0, x0, s0){
  tmp1 = c()
  tmp2 = c()
  
  prex = x0
  pres = s0
  
  ran_num1 = rnorm(n0* m0)
  ran_num2 = rho0 * ran_num1 + sqrt(1-rho0^2) * rnorm(n0* m0)
  
  del_tmp = del/m0
  for (i in 1:(n0*m0)){
    prex = prex + kappa0*(theta0-prex)*del_tmp+eta0* (del_tmp)**(1/2) * ran_num1[i]  
    tmp1[i] = prex
    pres = pres + (mu0 - 1/2 * sigma0^2) * del_tmp +sigma0 * ((del_tmp)**(1/2)) * ran_num2[i]
    tmp2[i] = pres
  } 
  return(cbind(c(x0,tmp1[(rep(1:n0)*m0)]), c(s0,tmp2[(rep(1:n0)*m0)])))
}


Multi_nonstnry_gnrtr<-function(del, n0, m0, mu0, sigma0, kappa0, theta0, eta0, rho0, x0, s0){
  tmp1 = c()
  tmp2 = c()
  
  prex = x0
  pres = s0
  
  ran_num1 = rnorm(n0* m0)
  ran_num2 = rho0 * ran_num1 + sqrt(1-rho0^2) * rnorm(n0* m0)
  
  del_tmp = del/m0
  for (i in 1:(n0*m0)){
    prex = prex + kappa0*(theta0-prex)*del_tmp+eta0* (del_tmp)**(1/2) * ran_num1[i]  
    tmp1[i] = prex
    pres = pres + (mu0 - 1/2 * sigma0^2) * del_tmp +sigma0 * ((del_tmp)**(1/2)) * ran_num2[i]
    tmp2[i] = pres
  } 
  return(cbind(c(x0,tmp1[(rep(1:n0)*m0)]), c(s0,tmp2[(rep(1:n0)*m0)])))
}


MLE_Multi <-function(St, Xt, delta){
  n = length(St)-1
  mhat = (St[n+1] - St[1])/n
  Shat.sq =  ( sum(diff(St)^2) - 2 * mhat * (St[n+1] - St[1]) + n * (mhat^2) )/ n
  phat = 1/ (n * sum((Xt[1:n])^2 ) - (sum(Xt[1:n]))^2 ) * (n * sum(Xt[2:(n+1)] * Xt[1:n]) - (Xt[n+1] -Xt[1]) * sum(Xt[1:n]) - (sum(Xt[1:n]) )^2  )
  qhat =  ((Xt[n+1] - Xt[1]) + sum(Xt[1:n]) - phat * sum(Xt[1:n]) )/n
  #Vhat.sq = 1/n * (Xt[n+1]^2 - Xt[1]^2 + (1+phat^2) * sum(Xt[1:n]^2) - 2* phat * sum(Xt[1:n] * Xt[2:(n+1)]) + n* qhat )
  Vhat.sq = 1/n * sum((Xt[2:(n+1)] - phat * Xt[1:n] - qhat)^2)
  Chat = 1/(n * sqrt(Vhat.sq) * sqrt(Shat.sq)) * (sum(Xt[2:(n+1)] * diff(St)) - phat * sum(Xt[1:n] * diff(St)) - mhat * (Xt[n+1]-Xt[1]) - mhat * (1-phat) * sum(Xt[1:n]) )
  
  sigmahat = sqrt(Shat.sq/ delta);sigmahat
  muhat = mhat/ delta + (1/2) * sigmahat^2;muhat
  kappahat = -log(phat)/delta;kappahat
  thetahat = qhat / (1- phat);thetahat
  etahat = sqrt(2 * kappahat * Vhat.sq / (1-phat^2));etahat
  rhohat = kappahat * Chat * sqrt(Vhat.sq) * sqrt(Shat.sq) / (etahat * sigmahat * (1-phat));rhohat
  return(c(muhat, sigmahat, kappahat, thetahat, etahat, rhohat))
}
