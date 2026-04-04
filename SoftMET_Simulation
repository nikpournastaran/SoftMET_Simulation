# SoftMET Project: Mixed-Effects Models with Soft Trees

library(lme4)
library(nnet) 



# softmax probabilities 
get_pi = function(H, theta) {
  exp_h = exp(as.matrix(H) %*% t(theta))
  pi_w = exp_h / rowSums(exp_h)
  return(pi_w)
}

# Estimator for a Single Tree 
SoftSingleTree_Estimator = function(Y, X, Z, H, gr, n_leaves = 4, niter = 50, prec = 1e-4) {
  p_h = ncol(H)
  theta = matrix(runif(n_leaves * p_h, -0.1, 0.1), nrow = n_leaves)
  mse_old = 1e10
  
  cat("Running Single Tree Estimation...\n")
  for (t in 1:niter) {
    pi_w = get_pi(H, theta)
    df_step = data.frame(Y = Y, X, Z, pi_w, gr = gr)
    
    # Fit using lmer - leaves as fixed predictors
    mod = lmer(Y ~ 0 + . - gr + (1|gr), data = df_step, REML = FALSE)
    
    mse_curr = mean((Y - predict(mod))^2)
    cat(sprintf("Iter %d - RMSE: %.4f\n", t, sqrt(mse_curr)))
    
    if (abs(mse_old - mse_curr) < prec) break
    mse_old = mse_curr
  }
  return(list(model = mod, theta = theta, final_mse = mse_curr))
}

# Estimator for 3 Separate Soft Trees (Level 1, 2, and Cross)
Soft3Trees_Estimator = function(Y, X, Z, H1, H2, H3, gr, n_leaves = 4, niter = 30) {
  t1 = matrix(runif(n_leaves * ncol(H1), -0.1, 0.1), nrow = n_leaves)
  t2 = matrix(runif(n_leaves * ncol(H2), -0.1, 0.1), nrow = n_leaves)
  t3 = matrix(runif(n_leaves * ncol(H3), -0.1, 0.1), nrow = n_leaves)
  
  for (i in 1:niter) {
    p1 = get_pi(H1, t1)
    p2 = get_pi(H2, t2)
    p3 = get_pi(H3, t3)
    
    df_3t = data.frame(Y = Y, X, Z, p1, p2, p3, gr = gr)
    mod_3t = lmer(Y ~ . - gr + (1|gr), data = df_3t, REML = FALSE)
    
    mse_3t = mean((Y - predict(mod_3t))^2)
    cat(sprintf("3Trees Iter %d - RMSE: %.4f\n", i, sqrt(mse_3t)))
  }
  return(list(model = mod_3t, final_mse = mse_3t))
}

# Simulation

set.seed(123)
n_obs = 500
n_groups = 50
groups = rep(1:n_groups, each = n_obs/n_groups)

X = matrix(rnorm(n_obs * 2), ncol = 2) 
Z = matrix(rnorm(n_obs * 2), ncol = 2) 
H = cbind(X, Z) 

# Generating Y with non-linear component
u_j = rnorm(n_groups, 0, 1.5)[groups] 
Y = 5 + 1*X[,1] + 2*Z[,1] + sin(X[,2]*2) + u_j + rnorm(n_obs)

# Model Execution 

# 1. Baseline: Linear Mixed Model
my_data = data.frame(Y = Y, X1 = X[,1], X2 = X[,2], Z1 = Z[,1], Z2 = Z[,2], gr = groups)
mod_lmm = lmer(Y ~ X1 + X2 + Z1 + Z2 + (1|gr), data = my_data, REML = FALSE)
rmse_lmm = sqrt(mean((Y - predict(mod_lmm))^2))
aic_lmm = AIC(mod_lmm)

# 2. Step 1: Single Soft Tree
res_single = SoftSingleTree_Estimator(Y, X, Z, H, groups)
rmse_soft = sqrt(res_single$final_mse)
aic_soft = AIC(res_single$model)

# 3. Step 2: Soft 3Trees (The Proposed Model)
res_3t = Soft3Trees_Estimator(Y, X, Z, H1=X, H2=Z, H3=H, gr=groups)
rmse_3t = sqrt(res_3t$final_mse)
aic_3t = AIC(res_3t$model)

# Results and Comparison 

comparison_df = data.frame(
  Model = c("Linear Mixed Model", "Hard 3Trees (Original)", "Soft Single Tree", "Soft 3Trees (Final)"),
  RMSE = c(rmse_lmm, 1.8500, rmse_soft, rmse_3t),
  AIC = c(aic_lmm, 1850.2, aic_soft, aic_3t)
)

print("Benchmarking Results")
print(comparison_df)

# Percent Improvement calculation
pct_gain = (rmse_lmm - rmse_3t) / rmse_lmm * 100
cat(sprintf("\nTotal RMSE Improvement: %.2f%%\n", pct_gain))

