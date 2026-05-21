library(lme4)

# Softmax 
get_pi <- function(H, th) {
  sc <- as.matrix(H) %*% t(th)
  sc <- sc - apply(sc, 1, max) # avoid overflow
  exp_h <- exp(sc)
  return(exp_h / rowSums(exp_h))
}

# Update a single tree using BFGS
upd_tree <- function(Y, H, th, n_l) {
  p_ncol <- ncol(H)
  
  # Drop first column to fix collinearity 
  get_mu <- function(p) { 
    m <- coef(lm(Y ~ p[, -1, drop = FALSE]))
    m[is.na(m)] <- 0
    return(as.numeric(m))
  }
  
  # MSE loss function
  obj <- function(v) {
    p <- get_pi(H, matrix(v, nrow = n_l, ncol = p_ncol, byrow = TRUE))
    m <- get_mu(p)
    yh <- m[1] + p[, -1, drop = FALSE] %*% m[-1]
    mean((Y - yh)^2)
  }
  
  opt <- optim(as.vector(t(th)), obj, method = "BFGS", control = list(maxit = 50))
  new_th <- matrix(opt$par, nrow = n_l, ncol = p_ncol, byrow = TRUE)
  return(list(th = new_th, pi = get_pi(H, new_th)))
}

# Generate data for scenario 1
gen_data <- function(n = 500, g = 50) {
  gr <- factor(rep(1:g, each = n/g))
  X1 <- rnorm(n); X2 <- rnorm(n)
  Z1 <- rnorm(g)[gr]; Z2 <- rnorm(g)[gr]
  Y <- 5 + X1 + Z1 + rnorm(g, 0, sqrt(3))[gr] + rnorm(n)
  return(data.frame(Y, X1, X2, Z1, Z2, gr))
}

# 3-tree backfitting and final inference
softmet_3trees <- function(d, n_leaves = 4, niter = 5) {
  H <- as.matrix(d[, c("X1", "X2", "Z1", "Z2")])
  p_ncol <- ncol(H)
  
  # Initialize 3 separate theta matrices
  th1 <- matrix(runif(n_leaves * p_ncol, -0.1, 0.1), nrow = n_leaves)
  th2 <- matrix(runif(n_leaves * p_ncol, -0.1, 0.1), nrow = n_leaves)
  th3 <- matrix(runif(n_leaves * p_ncol, -0.1, 0.1), nrow = n_leaves)
  
  # Initial tree predictions divided by 3 like the prof's code
  yhT1 <- yhT2 <- yhT3 <- mean(d$Y) / 3
  
  # Stage 1: Backfitting loop 
  for (i in 1:niter) {
    # Update linear part via lmer
    d$Y_resid <- d$Y - (yhT1 + yhT2 + yhT3)
    yhL <- fitted(lmer(Y_resid ~ X1 + X2 + Z1 + Z2 + (1|gr), data = d, REML = FALSE))
    
    # Update Tree 1
    Y_p1 <- d$Y - yhL - yhT2 - yhT3
    tr1 <- upd_tree(Y_p1, H, th1, n_leaves)
    th1 <- tr1$th
    yhT1 <- predict(lm(Y_p1 ~ tr1$pi[, -1, drop = FALSE]))
    
    # Update Tree 2
    Y_p2 <- d$Y - yhL - yhT1 - yhT3
    tr2 <- upd_tree(Y_p2, H, th2, n_leaves)
    th2 <- tr2$th
    yhT2 <- predict(lm(Y_p2 ~ tr2$pi[, -1, drop = FALSE]))
    
    # Update Tree 3
    Y_p3 <- d$Y - yhL - yhT1 - yhT2
    tr3 <- upd_tree(Y_p3, H, th3, n_leaves)
    th3 <- tr3$th
    yhT3 <- predict(lm(Y_p3 ~ tr3$pi[, -1, drop = FALSE]))
  }
  
  #  Stage 2: Final inference (Algorithm 1)
  # Get bases and drop first column for all 3 trees
  Phi1 <- as.data.frame(get_pi(H, th1)[, -1, drop = FALSE])
  Phi2 <- as.data.frame(get_pi(H, th2)[, -1, drop = FALSE])
  Phi3 <- as.data.frame(get_pi(H, th3)[, -1, drop = FALSE])
  
  names(Phi1) <- paste0("T1_V", 1:ncol(Phi1))
  names(Phi2) <- paste0("T2_V", 1:ncol(Phi2))
  names(Phi3) <- paste0("T3_V", 1:ncol(Phi3))
  
  # Combine data (augmented design matrix)
  Phi_all <- cbind(Phi1, Phi2, Phi3)
  D_final <- cbind(d[, c("Y", "X1", "X2", "Z1", "Z2", "gr")], Phi_all)
  
  # Fit base model and final softmet model
  m_base <- lmer(Y ~ X1 + X2 + Z1 + Z2 + (1|gr), data = D_final, REML = FALSE)
  
  f_trees <- paste(names(Phi_all), collapse = " + ")
  form_soft <- as.formula(paste("Y ~ X1 + X2 + Z1 + Z2 +", f_trees, "+ (1|gr)"))
  m_soft <- lmer(form_soft, data = D_final, REML = FALSE)
  
  return(list(base = m_base, soft = m_soft))
}

# --- Run simulation and test ---
my_data <- gen_data()
res <- softmet_3trees(my_data)

# Compare models using the prof's anova step
anova(res$base, res$soft)
