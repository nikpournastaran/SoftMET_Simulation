library(lme4)

# Softmax function 
get_pi <- function(H, th) {
  sc <- as.matrix(H) %*% t(th)
  sc <- sc - apply(sc, 1, max) # Subtract max to avoid exp overflow
  exp_h <- exp(sc)
  return(exp_h / rowSums(exp_h))
}

# Update theta using BFGS
upd_tree <- function(Y, H, th, n_l) {
  p_ncol <- ncol(H)
  
  # Estimate leaf values (mu) dropping first column for identifiability
  get_mu <- function(p) { 
    m <- coef(lm(Y ~ p[, -1, drop = FALSE]))
    m[is.na(m)] <- 0
    return(as.numeric(m))
  }
  
  # MSE objective function
  obj <- function(v) {
    p <- get_pi(H, matrix(v, nrow = n_l, ncol = p_ncol, byrow = TRUE))
    m <- get_mu(p)
    yh <- m[1] + p[, -1, drop = FALSE] %*% m[-1]
    mean((Y - yh)^2)
  }
  
  # Run R optimizer
  opt <- optim(as.vector(t(th)), obj, method = "BFGS", control = list(maxit = 50))
  new_th <- matrix(opt$par, nrow = n_l, ncol = p_ncol, byrow = TRUE)
  return(list(th = new_th, pi = get_pi(H, new_th)))
}

# Data generation for scenario 1
gen_data <- function(n = 500, g = 50) {
  gr <- factor(rep(1:g, each = n/g))
  X1 <- rnorm(n); X2 <- rnorm(n)
  Z1 <- rnorm(g)[gr]; Z2 <- rnorm(g)[gr]
  Y <- 5 + X1 + Z1 + rnorm(g, 0, sqrt(3))[gr] + rnorm(n)
  return(data.frame(Y, X1, X2, Z1, Z2, gr))
}

# Main two-stage SoftMET estimation procedure
softmet_estimate <- function(d, n_leaves = 4, niter = 5) {
  H <- as.matrix(d[, c("X1", "X2", "Z1", "Z2")])
  p_ncol <- ncol(H)
  th <- matrix(runif(n_leaves * p_ncol, -0.1, 0.1), nrow = n_leaves)
  yhT <- mean(d$Y)
  
  # Stage 1: Backfitting loop (structural selection)
  for (i in 1:niter) {
    d$Y_resid <- d$Y - yhT
    yhL <- fitted(lmer(Y_resid ~ X1 + X2 + Z1 + Z2 + (1|gr), data = d, REML = FALSE))
    tr <- upd_tree(d$Y - yhL, H, th, n_leaves)
    th <- tr$th
    yhT <- predict(lm((d$Y - yhL) ~ tr$pi[, -1, drop = FALSE]))
  }
  
  # Stage 2: Final Inference (the lmer step)
  Pi <- get_pi(H, th)
  Phi <- as.data.frame(Pi[, -1, drop = FALSE]) # Drop first column for identifiability
  
  # Construct augmented design matrix
  D_final <- cbind(d[, c("Y", "X1", "X2", "Z1", "Z2", "gr")], Phi)
  
  # Fit baseline and final softmet model on synchronized data
  m_base <- lmer(Y ~ X1 + X2 + Z1 + Z2 + (1|gr), data = D_final, REML = FALSE)
  
  f_trees <- paste(names(Phi), collapse = " + ")
  form_soft <- as.formula(paste("Y ~ X1 + X2 + Z1 + Z2 +", f_trees, "+ (1|gr)"))
  m_soft <- lmer(form_soft, data = D_final, REML = FALSE)
  
  return(list(base = m_base, soft = m_soft))
}

#  Model Comparison 
my_data <- gen_data()
res <- softmet_estimate(my_data)

#  ANOVA
anova(res$base, res$soft)
