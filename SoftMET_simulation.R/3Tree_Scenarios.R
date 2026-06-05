#SOFTMET


# Load libraries
library(lme4)
library(MASS)
library(knitr)
set.seed(2026)

# 1. Softmax Function
get_pi <- function(H, th) {
  sc <- as.matrix(H) %*% t(th)
  sc <- sc - apply(sc, 1, max)        # Numerical stability
  exp_h <- exp(sc)
  return(exp_h / rowSums(exp_h))
}

# 2. Update Tree Function 
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
  
  opt <- optim(as.vector(t(th)), obj, method = "BFGS", 
               control = list(maxit = 30))
  
  # Final predictions
  final_p <- get_pi(H, matrix(opt$par, nrow = n_l, ncol = p_ncol, byrow = TRUE))
  final_m <- get_mu(final_p)
  yhat <- final_m[1] + final_p[, -1, drop = FALSE] %*% final_m[-1]
  
  return(list(th = matrix(opt$par, nrow = n_l, ncol = p_ncol, byrow = TRUE),
              yhat = as.numeric(yhat)))
}

# 3. Data Generation

> # separate train and test datasets
> gen_data_split <- function(scenario = 1) {
+     g <- 50
+     n_j <- 15
+     n <- g * n_j
+     
+     generate_single_set <- function() {
+         gr <- factor(rep(1:g, each = n_j))
+         Sigma <- matrix(c(1.0, 0.4, 0.4, 1.0), 2, 2)
+         
+         X <- mvrnorm(n, mu = c(0, 0), Sigma = Sigma)
+         Z_unique <- mvrnorm(g, mu = c(0, 0), Sigma = Sigma)
+         
+         X1 <- X[,1]; X2 <- X[,2]
+         Z1 <- Z_unique[gr, 1]; Z2 <- Z_unique[gr, 2]
+         u_j <- rnorm(g, 0, sqrt(3))[gr]
+         eps <- rnorm(n, 0, 1)
+         
+         if(scenario == 1) Y <- 5 + X1 + Z1 + u_j + eps
+         if(scenario == 2) Y <- 5 + X1 + Z1 + 2*(X1>=0) + 3*(Z2<0) - 3*(X1<0 & Z2>=0) + u_j + eps
+         if(scenario == 3) Y <- 2*X1 + 4*Z1 + 2*X2^2 + 2*Z1*log(abs(X1)+0.01) + u_j + eps
+         
+         return(data.frame(Y, X1, X2, Z1, Z2, gr))
+     }
+     
+     return(list(train = generate_single_set(), test = generate_single_set()))
+ }


# 4. Main SoftMET Function 
softmet_3trees <- function(d,
                           n_leaves = 4,
                           niter = 50,
                           prec = 1e-4,
                           covLin = c("X1", "X2", "Z1", "Z2"),
                           covT1  = c("X1", "X2"),
                           covT2  = c("Z1", "Z2"),
                           covT3  = c("X1", "X2", "Z1", "Z2")) {
  
  Y <- d$Y
  gr <- d$gr
  
  XT1 <- as.matrix(d[, covT1, drop = FALSE])
  XT2 <- as.matrix(d[, covT2, drop = FALSE])
  XT3 <- as.matrix(d[, covT3, drop = FALSE])
  
  # Initialization
  YhatL  <- predict(lm(as.formula(paste("Y ~", paste(covLin, collapse = "+"))), data = d))
  YhatT1 <- YhatT2 <- YhatT3 <- mean(Y) / 3
  
  th1 <- matrix(runif(n_leaves * ncol(XT1), -0.1, 0.1), nrow = n_leaves)
  th2 <- matrix(runif(n_leaves * ncol(XT2), -0.1, 0.1), nrow = n_leaves)
  th3 <- matrix(runif(n_leaves * ncol(XT3), -0.1, 0.1), nrow = n_leaves)
  
  mse.best <- 1e8
  mse.train <- 1e8
  t <- 0
  d_conv <- 1
  
  best_th1 <- best_th2 <- best_th3 <- NULL
  
  # Backfitting Loop 
  while (d_conv != 0) {
    mse.train_old <- mse.train
    Y_residuals <- Y - YhatL - YhatT1 - YhatT2 - YhatT3
    
    # 1. Linear
    Y_pres <- Y_residuals + YhatL
    form_lin <- as.formula(paste("Y_pres ~", paste(covLin, collapse = "+"), "+ (1|gr)"))
    mod_l <- lmer(form_lin, data = d, REML = FALSE)
    YhatL <- fitted(mod_l)
    
    # 2. Tree 1
    Y_pres <- Y_pres - YhatL + YhatT1
    tr1 <- upd_tree(Y_pres, XT1, th1, n_leaves)
    th1 <- tr1$th
    YhatT1 <- tr1$yhat
    
    # 3. Tree 2
    Y_pres <- Y_pres - YhatT1 + YhatT2
    tr2 <- upd_tree(Y_pres, XT2, th2, n_leaves)
    th2 <- tr2$th
    YhatT2 <- tr2$yhat
    
    # 4. Tree 3
    Y_pres <- Y_pres - YhatT2 + YhatT3
    tr3 <- upd_tree(Y_pres, XT3, th3, n_leaves)
    th3 <- tr3$th
    YhatT3 <- tr3$yhat
    
    pred.final <- YhatL + YhatT1 + YhatT2 + YhatT3
    mse.train <- mean((Y - pred.final)^2)
    
    t <- t + 1
    d_conv <- (abs(mse.train_old - mse.train) > prec) * (t < niter)
    
    if (mse.train < mse.best) {
      mse.best <- mse.train
      best_th1 <- th1
      best_th2 <- th2
      best_th3 <- th3
    }
  }
  
  # Stage 2: Construct basis functions for train data
+     Phi1_tr <- as.data.frame(get_pi(XT1_tr, best_th1)[, -1, drop = FALSE])
+     Phi2_tr <- as.data.frame(get_pi(XT2_tr, best_th2)[, -1, drop = FALSE])
+     Phi3_tr <- as.data.frame(get_pi(XT3_tr, best_th3)[, -1, drop = FALSE])
+     
+     # Informative column names 
+     colnames(Phi1_tr) <- paste0("T1_leaf", 2:(ncol(Phi1_tr)+1))
+     colnames(Phi2_tr) <- paste0("T2_leaf", 2:(ncol(Phi2_tr)+1))
+     colnames(Phi3_tr) <- paste0("T3_leaf", 2:(ncol(Phi3_tr)+1))
+     
+     df_fin_tr <- cbind(d_train, Phi1_tr, Phi2_tr, Phi3_tr)
+     basis_names <- c(colnames(Phi1_tr), colnames(Phi2_tr), colnames(Phi3_tr))
+     
+     # Fit final models on training data
+     m_base <- lmer(Y ~ X1 + X2 + Z1 + Z2 + (1|gr), data = df_fin_tr, REML = FALSE)
+     
+     f_soft <- as.formula(paste("Y ~", paste(covLin, collapse = "+"), 
+                                "+", paste(basis_names, collapse = " + "), 
+                                "+ (1|gr)"))
+     m_soft <- lmer(f_soft, data = df_fin_tr, REML = FALSE)
+     
+     # Process test data using the optimized train routing parameters
+     XT1_te <- as.matrix(d_test[, covT1, drop = FALSE])
+     XT2_te <- as.matrix(d_test[, covT2, drop = FALSE])
+     XT3_te <- as.matrix(d_test[, covT3, drop = FALSE])
+     
+     Phi1_te <- as.data.frame(get_pi(XT1_te, best_th1)[, -1, drop = FALSE])
+     Phi2_te <- as.data.frame(get_pi(XT2_te, best_th2)[, -1, drop = FALSE])
+     Phi3_te <- as.data.frame(get_pi(XT3_te, best_th3)[, -1, drop = FALSE])
+     
+     colnames(Phi1_te) <- colnames(Phi1_tr)
+     colnames(Phi2_te) <- colnames(Phi2_tr)
+     colnames(Phi3_te) <- colnames(Phi3_tr)
+     
+     df_fin_te <- cbind(d_test, Phi1_te, Phi2_te, Phi3_te)
+     
+     return(list(base = m_base, soft = m_soft, test_data = df_fin_te))
+ }


# 5. Evaluation

cat("\n--- Single Run ANOVA Detailed Comparison ---\n")
results_anova <- lapply(1:3, function(sc) {
  fit <- softmet_3trees(gen_data(scenario = sc))
  res <- anova(fit$base, fit$soft)
  
  data.frame(
    Scenario = sc,
    AIC_Base = round(res$AIC[1], 2),
    AIC_Soft = round(res$AIC[2], 2),
    BIC_Soft = round(res$BIC[2], 2),
    LogLik_Soft = round(res$logLik[2], 2),
    P_Val = format.pval(res$`Pr(>Chisq)`[2], eps = 0.001, digits = 3)
  )
})

print(kable(do.call(rbind, results_anova),
            align = "c",
            caption = "Model Comparison Metrics per Scenario "))

cat("\n--- Monte Carlo Simulation (10 Iterations) ---\n")
mc_bench <- lapply(1:3, function(sc) {
  mse_b <- mse_s <- sig <- numeric(10)
  for(i in 1:10) {
    f <- softmet_3trees(gen_data(scenario = sc))
    mse_b[i] <- mean(residuals(f$base)^2)
    mse_s[i] <- mean(residuals(f$soft)^2)
    sig[i]   <- anova(f$base, f$soft)$`Pr(>Chisq)`[2] < 0.05
  }
  data.frame(Scenario = sc, 
             Power = mean(sig), 
             MSE_Base = mean(mse_b), 
             MSE_Soft = mean(mse_s), 
             Imp = mean(mse_b - mse_s))
})

print(kable(do.call(rbind, mc_bench), digits = 3))
