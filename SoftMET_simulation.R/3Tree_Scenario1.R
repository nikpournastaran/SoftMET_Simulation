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


# Softmet 3-Trees Backfitting Algorithm (Prof's Structure)
softmet_3trees <- function(Y, X, gr, covLin=NULL, covT1=NULL, covT2=NULL, covT3=NULL,
                           niter = 50, re_form = "(1|gr)", n_leaves = 4, prec=1e-4, extended=FALSE) {
  
  # Input validation
  if (NROW(gr) != NROW(X)) {
    stop("Invalid input: gr must have the same length/rows as the number of rows in X.")
  }
  
  if (!grepl("^\\(.*\\)$", re_form)) {
    re_form <- paste0("(", re_form, ")")
  }
  
  # Settings
  library(lme4)
  
  if(!is.null(covLin)) covLin  <- paste0(covLin , collapse="+")
  if(is.null(covLin)) covLin  <- "1"
  
  mydata <- data.frame(Y, X, gr)
  p <- ncol(X)
  n <- nrow(X)
  
  # Prepare Matrix H for Softmax trees based on provided covariates
  all_covs <- unique(c(covT1, covT2, covT3))
  if(is.null(all_covs)) {
    H <- as.matrix(X)
  } else {
    H <- as.matrix(X[, all_covs, drop=FALSE])
  }
  p_ncol <- ncol(H)
  
  myform1 <- as.formula(paste0("Y_pres ~ ", covLin, " + ", re_form))
  
  #initialization 
  YhatT1 <- YhatT2 <- YhatT3 <- mean(Y)/3
  YhatL <- predict(lm(as.formula(paste0("Y ~ ", covLin)), data=mydata))
  
  # Initialize 3 separate theta matrices for Softmax
  th1 <- matrix(runif(n_leaves * p_ncol, -0.1, 0.1), nrow = n_leaves)
  th2 <- matrix(runif(n_leaves * p_ncol, -0.1, 0.1), nrow = n_leaves)
  th3 <- matrix(runif(n_leaves * p_ncol, -0.1, 0.1), nrow = n_leaves)
  
  msetrain <- mse.best <- 10000000
  t <- 0 
  d <- 1
  
  #### Start iterative procedure
  while (d != 0){
    msetrain_old <- msetrain
    
    ##### Compute residuals
    Y_residuals <- Y - YhatL - YhatT1 - YhatT2 - YhatT3
    
    ### Linear part
    Y_pres <- Y_residuals + YhatL
    mydata$Y_pres <- Y_pres
    mod1 <- suppressWarnings(lmer(myform1, REML=FALSE, data=mydata, verbose = 0))
    YhatL <- predict(mod1, re.form = NULL, random.only=FALSE, type="response")
    
    ### fit tree T1
    Y_pres <- Y_pres - YhatL + YhatT1
    YhatT1 <- rep(0,n); tr1 <- NULL
    if(!is.null(covT1)){
      H1 <- as.matrix(X[, covT1, drop=FALSE])
      tr1 <- upd_tree(Y_pres, H1, th1, n_leaves)
      YhatT1 <- predict(lm(Y_pres ~ tr1$pi[, -1, drop = FALSE]))
    }
    
    ### fit tree T2
    Y_pres <- Y_pres - YhatT1 + YhatT2
    YhatT2 <- rep(0,n); tr2 <- NULL
    if(!is.null(covT2)){
      H2 <- as.matrix(X[, covT2, drop=FALSE])
      tr2 <- upd_tree(Y_pres, H2, th2, n_leaves)
      YhatT2 <- predict(lm(Y_pres ~ tr2$pi[, -1, drop = FALSE]))
    }
    
    ### fit tree T3
    Y_pres <- Y_pres - YhatT2 + YhatT3
    YhatT3 <- rep(0,n); tr3 <- NULL
    if(!is.null(covT3)){
      H3 <- as.matrix(X[, covT3, drop=FALSE])
      tr3 <- upd_tree(Y_pres, H3, th3, n_leaves)
      YhatT3 <- predict(lm(Y_pres ~ tr3$pi[, -1, drop = FALSE]))
    }
    
    pred.final <- YhatL + YhatT1 + YhatT2 + YhatT3
    msetrain <- mean((Y-pred.final)^2)
    
    t=t+1
    cat(paste0(rep('=', t), collapse = ''))
    d <- (abs(msetrain_old - msetrain)> prec)*(t < niter)
    
    # update best trees
    if (msetrain < mse.best) {
      mse.best <- msetrain
      th1.best <- if(!is.null(tr1)) tr1$th else th1
      th2.best <- if(!is.null(tr2)) tr2$th else th2
      th3.best <- if(!is.null(tr3)) tr3$th else th3
    }
  }#### End iterative procedure
  
  #########################
  # construct final model matrix
  mydata2 <- as.data.frame(cbind(Y, X, gr))
  
  f_trees <- c()
  if(!is.null(covT1)){
    H1 <- as.matrix(X[, covT1, drop=FALSE])
    Phi1 <- as.data.frame(get_pi(H1, th1.best)[, -1, drop = FALSE])
    names(Phi1) <- paste0("T1_V", 1:ncol(Phi1))
    mydata2 <- cbind(mydata2, Phi1)
    f_trees <- c(f_trees, names(Phi1))
  }
  if(!is.null(covT2)){
    H2 <- as.matrix(X[, covT2, drop=FALSE])
    Phi2 <- as.data.frame(get_pi(H2, th2.best)[, -1, drop = FALSE])
    names(Phi2) <- paste0("T2_V", 1:ncol(Phi2))
    mydata2 <- cbind(mydata2, Phi2)
    f_trees <- c(f_trees, names(Phi2))
  }
  if(!is.null(covT3)){
    H3 <- as.matrix(X[, covT3, drop=FALSE])
    Phi3 <- as.data.frame(get_pi(H3, th3.best)[, -1, drop = FALSE])
    names(Phi3) <- paste0("T3_V", 1:ncol(Phi3))
    mydata2 <- cbind(mydata2, Phi3)
    f_trees <- c(f_trees, names(Phi3))
  }
  
  # Formulate final formula
  myform3 <- paste0("Y ~ ", covLin)
  if(covLin == "1") myform3 <- "Y ~ "
  if(length(f_trees) > 0) myform3 <- paste0(myform3, " + ", paste(f_trees, collapse = " + "))
  myform3 <- as.formula(paste0(myform3, "+", re_form))
  
  cat("\n", "Estimating the final model...","\n")
  mod.final <- suppressMessages(lmer(myform3, REML=FALSE, data=mydata2, verbose = 0))
  
  mse.final <- mean((Y - predict(mod.final, re.form = ~0, random.only=FALSE, type="response"))^2)
  mse.random  <- mean((Y - predict(mod.final, re.form = NULL, random.only=FALSE, type="response"))^2)
  
  myris <- list(mse.final=mse.final, mse.random=mse.random, mod.final= mod.final)
  if(!is.null(covT1)) myris$th1 <- th1.best
  if(!is.null(covT2)) myris$th2 <- th2.best
  if(!is.null(covT3)) myris$th3 <- th3.best
  if(extended) myris$modmat <- mydata2
  return(myris)
}

# --- Run simulation and test ---
sim_data <- gen_data()

Y_val <- sim_data$Y
X_mat <- sim_data[, c("X1", "X2", "Z1", "Z2")]
gr_val <- sim_data$gr

# Calling our function exactly like the prof's syntax
res <- softmet_3trees(Y = Y_val, 
                      X = X_mat, 
                      gr = gr_val, 
                      covLin = c("X1", "X2", "Z1", "Z2"),
                      covT1 = c("X1", "X2"), 
                      covT2 = c("Z1", "Z2"), 
                      covT3 = c("X1", "Z2"),
                      niter = 5)

# Review final summary
summary(res$mod.final)




















































































































  
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

#  Run simulation and test 
my_data <- gen_data()
res <- softmet_3trees(my_data)

# Compare models using the prof's anova step
anova(res$base, res$soft)
