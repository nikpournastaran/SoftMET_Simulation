library(lme4)

# 1. Softmax with Numerical Stability 
get_pi <- function(H, th) {
  sc <- as.matrix(H) %*% t(th)
  sc <- sc - apply(sc, 1, max) # Subtract row_size max for stability
  exp_h <- exp(sc)
  return(exp_h / rowSums(exp_h))
}

# 2. Update Tree Parameters 
upd_tree <- function(Y, H, th, n_l) {
  # Helper to estimate leaf values (mu)
  get_mu <- function(p) { 
    m <- coef(lm(Y ~ p - 1))
    m[is.na(m)] <- 0
    return(as.numeric(m))
  }
  # Objective function: MSE
  obj <- function(v) {
    p <- get_pi(H, matrix(v, n_l))
    mean((Y - p %*% get_mu(p))^2)
  }
  # BFGS optimization to update theta
  opt <- optim(as.vector(th), obj, method = "BFGS", control = list(maxit = 50))
  new_th <- matrix(opt$par, n_l)
  p_f <- get_pi(H, new_th)
  return(list(th = new_th, mu = get_mu(p_f), pi = p_f))
}

#  3. Data Generation - Scenario 1 
gen_data <- function(n = 500, g = 50) {
  gr <- factor(rep(1:g, each = n/g))
  X1 <- rnorm(n); X2 <- rnorm(n) # Level 1
  Z1 <- rnorm(g)[gr]; Z2 <- rnorm(g)[gr] # Level 2
  u_j <- rnorm(g, 0, sqrt(3))[gr] # Random Effect
  
  # Scenario 1: Y = 5 + 1*X1 + 1*Z1 + u_j + eps
  Y <- 5 + X1 + Z1 + u_j + rnorm(n)
  return(data.frame(Y, X1, X2, Z1, Z2, gr))
}

#  4. Backfitting
SoftTree_Estimator <- function(d, n_leaves = 4, niter = 10) {
  H <- as.matrix(d[, c("X1", "X2", "Z1", "Z2")])
  th <- matrix(runif(n_leaves * ncol(H), -0.1, 0.1), n_leaves)
  yhT <- mean(d$Y) # Initial tree prediction
  
  for (i in 1:niter) {
    # Step 1 & 2: Estimate Mixed Model (lmer)
    d$Y_resid <- d$Y - yhT
    mod_lmm <- lmer(Y_resid ~ X1 + X2 + Z1 + Z2 + (1|gr), data = d, REML = FALSE)
    yhL <- fitted(mod_lmm)
    
    # Step 3: Update Tree Theta
    tr <- upd_tree(d$Y - yhL, H, th, n_leaves)
    th <- tr$th
    yhT <- tr$pi %*% tr$mu
  }
  return(yhL + yhT) # Final predictions
}

# 5. Simulation
run_sim <- function(reps = 100) {
  cat("Simulation started...\n")
  res <- replicate(reps, {
    d <- gen_data()
    
    # Baseline: Standard LMM
    m_base <- lmer(Y ~ X1 + X2 + Z1 + Z2 + (1|gr), data = d, REML = FALSE)
    rmse_lmm <- sqrt(mean(residuals(m_base)^2))
    
    # Proposed: SoftTree Hybrid
    y_hat_soft <- SoftTree_Estimator(d)
    rmse_soft <- sqrt(mean((d$Y - y_hat_soft)^2))
    
    return(c(LMM = rmse_lmm, SoftTree = rmse_soft))
  })
  
  final_res <- rowMeans(res)
  return(final_res)
}

# Execute
print(run_sim(100))
