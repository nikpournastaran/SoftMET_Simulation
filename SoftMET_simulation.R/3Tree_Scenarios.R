# load libraries
library(lme4)
library(MASS)
library(knitr)
set.seed(2026)



softmet_3trees <- function(d, 
                           n_leaves = 4, 
                           niter = 50, 
                           prec = 1e-4,
                           covLin = c("X1", "X2", "Z1", "Z2"),
                           covT1  = c("X1", "X2"),      
                           covT2  = c("Z1", "Z2"),    
                           covT3  = c("X1", "X2", "Z1", "Z2")) 
  
 
  Y  <- d$Y
  gr <- d$gr
  n  <- nrow(d)
  
  XLin <- as.matrix(d[, covLin, drop = FALSE])
  XT1  <- as.matrix(d[, covT1,  drop = FALSE])
  XT2  <- as.matrix(d[, covT2,  drop = FALSE])
  XT3  <- as.matrix(d[, covT3,  drop = FALSE])
  
=
  YhatL  <- predict(lm(as.formula(paste("Y ~", paste(covLin, collapse = "+"))), data = d))
  YhatT1 <- YhatT2 <- YhatT3 <- mean(Y) / 3
  
  p_ncol1 <- ncol(XT1); p_ncol2 <- ncol(XT2); p_ncol3 <- ncol(XT3)
  
  th1 <- matrix(runif(n_leaves * p_ncol1, -0.1, 0.1), nrow = n_leaves)
  th2 <- matrix(runif(n_leaves * p_ncol2, -0.1, 0.1), nrow = n_leaves)
  th3 <- matrix(runif(n_leaves * p_ncol3, -0.1, 0.1), nrow = n_leaves)
  
  mse.best <- 1e8
  mse.train <- 1e8
  t <- 0
  d_conv <- 1
  
  best_th1 <- best_th2 <- best_th3 <- NULL
  best_YhatL <- best_YhatT1 <- best_YhatT2 <- best_YhatT3 <- NULL
  
  # BACKFITTING LOOP 
  while (d_conv != 0) {
    mse.train_old <- mse.train
    
    Y_residuals <- Y - YhatL - YhatT1 - YhatT2 - YhatT3
    
    #1. Linear Part 
    Y_pres <- Y_residuals + YhatL
    form_lin <- as.formula(paste("Y_pres ~", paste(covLin, collapse = "+"), "+ (1|gr)"))
    mod_l <- lmer(form_lin, data = d, REML = FALSE)
    YhatL <- fitted(mod_l)
    
    #  2. Tree 1
    Y_pres <- Y_pres - YhatL + YhatT1
    tr1 <- upd_tree(Y_pres, XT1, th1, n_leaves)
    th1 <- tr1$th
    YhatT1 <- tr1$yhat
    
    #  3. Tree 2 
    Y_pres <- Y_pres - YhatT1 + YhatT2
    tr2 <- upd_tree(Y_pres, XT2, th2, n_leaves)
    th2 <- tr2$th
    YhatT2 <- tr2$yhat
    
    #  4. Tree 3
    Y_pres <- Y_pres - YhatT2 + YhatT3
    tr3 <- upd_tree(Y_pres, XT3, th3, n_leaves)
    th3 <- tr3$th
    YhatT3 <- tr3$yhat
    
    # MSE
    pred.final <- YhatL + YhatT1 + YhatT2 + YhatT3
    mse.train <- mean((Y - pred.final)^2)
    
    t <- t + 1
    d_conv <- (abs(mse.train_old - mse.train) > prec) * (t < niter)
    
    if (mse.train < mse.best) {
      mse.best <- mse.train
      best_th1 <- th1; best_th2 <- th2; best_th3 <- th3
      best_YhatL <- YhatL; best_YhatT1 <- YhatT1
      best_YhatT2 <- YhatT2; best_YhatT3 <- YhatT3
    }
  }
  
  # STAGE 2: Final Model with Soft Basis 
  Phi1 <- get_pi(XT1, best_th1)[, -1, drop = FALSE]
  Phi2 <- get_pi(XT2, best_th2)[, -1, drop = FALSE]
  Phi3 <- get_pi(XT3, best_th3)[, -1, drop = FALSE]
  
  df_fin <- cbind(d, Phi1, Phi2, Phi3)
  
  #  Basis
  n_basis <- ncol(Phi1) + ncol(Phi2) + ncol(Phi3)
  colnames(df_fin)[(ncol(d)+1):ncol(df_fin)] <- paste0("Basis_", 1:n_basis)
  
  basis_names <- colnames(df_fin)[(ncol(d)+1):ncol(df_fin)]
  
  # 
  m_base <- lmer(Y ~ X1 + X2 + Z1 + Z2 + (1|gr), data = df_fin, REML = FALSE)
  
  # 
  f_soft <- as.formula(paste("Y ~", paste(covLin, collapse = "+"), 
                             "+", paste(basis_names, collapse = " + "), 
                             "+ (1|gr)"))
  m_soft <- lmer(f_soft, data = df_fin, REML = FALSE)
  
  return(list(base = m_base, 
              soft = m_soft,
              th1 = best_th1, th2 = best_th2, th3 = best_th3,
              mse.best = mse.best,
              niter = t))
}

# Update routing parameters using BFGS
upd_tree <- function(Y, H, th, n_l) {
  p_ncol <- ncol(H)
  
  get_mu <- function(p) { 
    m <- coef(lm(Y ~ p[, -1, drop = FALSE]))
    m[is.na(m)] <- 0
    return(as.numeric(m))
  }
  
  obj <- function(v) {
    p <- get_pi(H, matrix(v, nrow = n_l, ncol = p_ncol, byrow = TRUE))
    m <- get_mu(p)
    yh <- m[1] + p[, -1, drop = FALSE] %*% m[-1]
    mean((Y - yh)^2)
  }
  
  opt <- optim(as.vector(t(th)), obj, method = "BFGS", control = list(maxit = 30))
  
  # Final Prediction
  final_p <- get_pi(H, matrix(opt$par, nrow = n_l, ncol = p_ncol, byrow = TRUE))
  final_m <- get_mu(final_p)
  yhat <- final_m[1] + final_p[, -1, drop = FALSE] %*% final_m[-1]
  
  return(list(th = matrix(opt$par, nrow = n_l, ncol = p_ncol, byrow = TRUE),
              yhat = as.numeric(yhat)))
}

#  Data Generation

gen_data <- function(n = 500, g = 50, scenario = 1) {
  gr <- factor(rep(1:g, each = n/g))
  Sigma <- matrix(c(1.0, 0.4, 0.4, 1.0), 2, 2)
  X <- mvrnorm(n, mu = c(0, 0), Sigma = Sigma)
  X1 <- X[,1]; X2 <- X[,2]
  Z_unique <- mvrnorm(g, mu = c(0, 0), Sigma = Sigma)
  Z1 <- Z_unique[gr, 1]; Z2 <- Z_unique[gr, 2]
  u_j <- rnorm(g, 0, sqrt(3))[gr]
  eps <- rnorm(n, 0, 1)
  
  # Scenarios: 1=Linear, 2=Quasi-linear, 3=Non-linear
  if(scenario == 1) Y <- 5 + X1 + Z1 + u_j + eps
  if(scenario == 2) Y <- 5 + X1 + Z1 + 2*(X1>=0) + 3*(Z2<0) - 3*(X1<0 & Z2>=0) + u_j + eps
  if(scenario == 3) Y <- 2*X1 + 4*Z1 + 2*X2^2 + 2*Z1*log(abs(X1)+0.01) + u_j + eps
  
  return(data.frame(Y, X1, X2, Z1, Z2, gr))
}

# SoftMET algorithm with 3-tree backfitting
softmet_3trees <- function(d, n_leaves = 4, niter = 5) {
  X_vars <- as.matrix(d[, c("X1", "X2", "Z1", "Z2")])
  p_ncol <- ncol(X_vars)
  th1 <- th2 <- th3 <- matrix(runif(n_leaves * p_ncol, -0.1, 0.1), nrow = n_leaves)
  yhT1 <- yhT2 <- yhT3 <- mean(d$Y)/3
  
  for (i in 1:niter) {
    # Update linear part
    d$Y_res <- d$Y - (yhT1 + yhT2 + yhT3)
    mod_l <- lmer(Y_res ~ X1 + X2 + Z1 + Z2 + (1|gr), data = d, REML = FALSE)
    yhL <- fitted(mod_l)
    
    # Backfitting tree components
    tr1 <- upd_tree(d$Y-yhL-yhT2-yhT3, X_vars, th1, n_leaves); th1 <- tr1$th
    yhT1 <- predict(lm((d$Y-yhL-yhT2-yhT3) ~ get_pi(X_vars, th1)[,-1]))
    
    tr2 <- upd_tree(d$Y-yhL-yhT1-yhT3, X_vars, th2, n_leaves); th2 <- tr2$th
    yhT2 <- predict(lm((d$Y-yhL-yhT1-yhT3) ~ get_pi(X_vars, th2)[,-1]))
    
    tr3 <- upd_tree(d$Y-yhL-yhT1-yhT2, X_vars, th3, n_leaves); th3 <- tr3$th
    yhT3 <- predict(lm((d$Y-yhL-yhT1-yhT2) ~ get_pi(X_vars, th3)[,-1]))
  }
  
  # Stage 2: Final Inference using soft-leaf basis
  Phi1 <- get_pi(X_vars, th1)[,-1]; Phi2 <- get_pi(X_vars, th2)[,-1]; Phi3 <- get_pi(X_vars, th3)[,-1]
  df_fin <- cbind(d[,1:6], Phi1, Phi2, Phi3)
  colnames(df_fin)[7:ncol(df_fin)] <- paste0("Basis_", 1:(3*(n_leaves-1)))
  
  m_base <- lmer(Y ~ X1 + X2 + Z1 + Z2 + (1|gr), data = df_fin, REML = FALSE)
  f_soft <- as.formula(paste("Y ~ X1+X2+Z1+Z2 +", paste(colnames(df_fin)[7:ncol(df_fin)], collapse="+"), "+ (1|gr)"))
  m_soft <- lmer(f_soft, data = df_fin, REML = FALSE)
  
  return(list(base = m_base, soft = m_soft))
}



# 1. Single Run ANOVA Comparison
cat("\n--- Single Run ANOVA Detailed Comparison ---\n")
results_anova <- lapply(1:3, function(sc) {
  # Generate data and fit models for one realization
  fit <- softmet_3trees(gen_data(scenario = sc))
  res <- anova(fit$base, fit$soft)
  
  # Extracting full metrics
  data.frame(
    Scenario = sc,
    AIC_Base = round(res$AIC[1], 2),
    AIC_Soft = round(res$AIC[2], 2),
    BIC_Soft = round(res$BIC[2], 2),       
    LogLik_Soft = round(res$logLik[2], 2), 
    # Formatting P-value for academic reporting
    P_Val = format.pval(res$`Pr(>Chisq)`[2], eps = 0.001, digits = 3)
  )
})

# Display the table using kable for a professional look
print(kable(do.call(rbind, results_anova), 
            align = "c", 
            caption = "Model Comparison Metrics per Scenario "))

# 2. Monte Carlo Simulation
cat("\n--- Monte Carlo Simulation (10 Iterations) ---\n")
mc_bench <- lapply(1:3, function(sc) {
  mse_b <- mse_s <- sig <- numeric(10)
  for(i in 1:10) {
    f <- softmet_3trees(gen_data(scenario = sc))
    mse_b[i] <- mean(residuals(f$base)^2); mse_s[i] <- mean(residuals(f$soft)^2)
    sig[i] <- anova(f$base, f$soft)$`Pr(>Chisq)`[2] < 0.05
  }
  data.frame(Scenario=sc, Power=mean(sig), MSE_Base=mean(mse_b), MSE_Soft=mean(mse_s), Imp=mean(mse_b-mse_s))
})
print(kable(do.call(rbind, mc_bench), digits=3))
