library(lme4)
> library(MASS)
> library(knitr)
> set.seed(2026)
> 
> # Softmax 
> get_pi <- function(H, th) {
+     sc <- as.matrix(H) %*% t(th)
+     sc <- sc - apply(sc, 1, max) # numerical stability
+     exp_h <- exp(sc)
+     return(exp_h / rowSums(exp_h))
+ }
> 
> # Optimizes via BFGS
> upd_tree <- function(Y, H, th, n_l) {
+     p_ncol <- ncol(H)
+     
+     # Drop first column to fix collinearity 
+     get_mu <- function(p) {
+         m <- coef(lm(Y ~ p[, -1, drop = FALSE]))
+         m[is.na(m)] <- 0
+         return(as.numeric(m))
+     }
+     
+     obj <- function(v) {
+         p <- get_pi(H, matrix(v, nrow = n_l, ncol = p_ncol, byrow = TRUE))
+         m <- get_mu(p)
+         yh <- m[1] + p[, -1, drop = FALSE] %*% m[-1]
+         mean((Y - yh)^2)
+     }
+     
+     opt <- optim(as.vector(t(th)), obj, method = "BFGS", control = list(maxit = 30))
+     
+     final_p <- get_pi(H, matrix(opt$par, nrow = n_l, ncol = p_ncol, byrow = TRUE))
+     final_m <- get_mu(final_p)
+     yhat <- final_m[1] + final_p[, -1, drop = FALSE] %*% final_m[-1]
+     
+     return(list(th = matrix(opt$par, nrow = n_l, ncol = p_ncol, byrow = TRUE),
+                 yhat = as.numeric(yhat)))
+ }
> 
> # Generate dataset split into train and test
> gen_data <- function(g = 50, n_j = 15, scenario = 1) {
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
+         data.frame(Y, X1, X2, Z1, Z2, gr)
+     }
+     
+     list(train = generate_single_set(), test = generate_single_set())
+ }
> 
> # Main SoftMET function for 3-tree backfitting estimation
> softmet_3trees <- function(d,
+                            n_leaves = 4,
+                            niter = 50,
+                            prec = 1e-4,
+                            covLin = c("X1", "X2", "Z1", "Z2"),
+                            covT1  = c("X1", "X2"),
+                            covT2  = c("Z1", "Z2"),
+                            covT3  = c("X1", "X2", "Z1", "Z2")) {
+     
+     Y <- d$Y
+     gr <- d$gr
+     
+     XT1 <- as.matrix(d[, covT1, drop = FALSE])
+     XT2 <- as.matrix(d[, covT2, drop = FALSE])
+     XT3 <- as.matrix(d[, covT3, drop = FALSE])
+     
+     # Initialization
+     YhatL  <- predict(lm(as.formula(paste("Y ~", paste(covLin, collapse = "+"))), data = d))
+     YhatT1 <- YhatT2 <- YhatT3 <- mean(Y) / 3
+     
+     th1 <- matrix(runif(n_leaves * ncol(XT1), -0.1, 0.1), nrow = n_leaves)
+     th2 <- matrix(runif(n_leaves * ncol(XT2), -0.1, 0.1), nrow = n_leaves)
+     th3 <- matrix(runif(n_leaves * ncol(XT3), -0.1, 0.1), nrow = n_leaves)
+     
+     mse.best <- 1e8
+     mse.train <- 1e8
+     t <- 0
+     d_conv <- 1
+     best_th1 <- best_th2 <- best_th3 <- NULL
+     
+     # Stage 1: Backfitting Loop 
+     while (d_conv != 0) {
+         mse.train_old <- mse.train
+         Y_residuals <- Y - YhatL - YhatT1 - YhatT2 - YhatT3
+         
+         # 1. Linear part update
+         Y_pres <- Y_residuals + YhatL
+         form_lin <- as.formula(paste("Y_pres ~", paste(covLin, collapse = "+"), "+ (1|gr)"))
+         mod_l <- lmer(form_lin, data = d, REML = FALSE)
+         YhatL <- fitted(mod_l)
+         
+         # 2. Tree 1 update
+         Y_pres <- Y_pres - YhatL + YhatT1
+         tr1 <- upd_tree(Y_pres, XT1, th1, n_leaves)
+         th1 <- tr1$th
+         YhatT1 <- tr1$yhat
+         
+         # 3. Tree 2 update (FIXED TYPO HERE)
+         Y_pres <- Y_pres - YhatT1 + YhatT2
+         tr2 <- upd_tree(Y_pres, XT2, th2, n_leaves)
+         th2 <- tr2$th
+         YhatT2 <- tr2$yhat
+         
+         # 4. Tree 3 update
+         Y_pres <- Y_pres - YhatT2 + YhatT3
+         tr3 <- upd_tree(Y_pres, XT3, th3, n_leaves)
+         th3 <- tr3$th
+         YhatT3 <- tr3$yhat
+         
+         pred.final <- YhatL + YhatT1 + YhatT2 + YhatT3
+         mse.train <- mean((Y - pred.final)^2)
+         
+         t <- t + 1
+         d_conv <- (abs(mse.train_old - mse.train) > prec) * (t < niter)
+         
+         if (mse.train < mse.best) {
+             mse.best <- mse.train
+             best_th1 <- th1
+             best_th2 <- th2
+             best_th3 <- th3
+         }
+     }
+     
+     # Stage 2
+     Phi1 <- as.data.frame(get_pi(XT1, best_th1)[, -1, drop = FALSE])
+     Phi2 <- as.data.frame(get_pi(XT2, best_th2)[, -1, drop = FALSE])
+     Phi3 <- as.data.frame(get_pi(XT3, best_th3)[, -1, drop = FALSE])
+     
+     colnames(Phi1) <- paste0("T1_leaf", 2:(ncol(Phi1) + 1))
+     colnames(Phi2) <- paste0("T2_leaf", 2:(ncol(Phi2) + 1))
+     colnames(Phi3) <- paste0("T3_leaf", 2:(ncol(Phi3) + 1))
+     
+     Phi_basis <- cbind(Phi1, Phi2, Phi3)
+     df_fin <- cbind(d, Phi_basis)
+     basis_names <- colnames(Phi_basis)
+     
+     # Fit final baseline and SoftMET models
+     m_base <- lmer(Y ~ X1 + X2 + Z1 + Z2 + (1|gr), data = df_fin, REML = FALSE)
+     f_soft <- reformulate(c(covLin, basis_names, "(1|gr)"), response = "Y")
+     m_soft <- lmer(f_soft, data = df_fin, REML = FALSE)
+     
+     return(list(base = m_base, 
+                 soft = m_soft, 
+                 best_th1 = best_th1,
+                 best_th2 = best_th2,
+                 best_th3 = best_th3))
+ }
> 
> # 5. Simulation 
> cat("\n Running Monte Carlo Simulation (100 Reps) \n")

 Running Monte Carlo Simulation (100 Reps) 
> 
> # For quick test use n_mc <- 5 or 10, set to 100 for final results
> n_mc <- 10 
> 
> mc_results <- lapply(1:3, function(sc) {
+     mse_b <- mse_s <- sig <- numeric(n_mc)
+     
+     for(i in 1:n_mc) {
+         dat   <- gen_data(scenario = sc)     
+         train <- dat$train
+         test  <- dat$test
+         
+         # Fit model on training data
+         f <- softmet_3trees(train)           
+         
+         # Construct the exact same basis functions for test data
+         XT1_te <- as.matrix(test[, c("X1", "X2"), drop = FALSE])
+         XT2_te <- as.matrix(test[, c("Z1", "Z2"), drop = FALSE])
+         XT3_te <- as.matrix(test[, c("X1", "X2", "Z1", "Z2"), drop = FALSE])
+         
+         Phi1_te <- as.data.frame(get_pi(XT1_te, f$best_th1)[, -1, drop = FALSE])
+         Phi2_te <- as.data.frame(get_pi(XT2_te, f$best_th2)[, -1, drop = FALSE])
+         Phi3_te <- as.data.frame(get_pi(XT3_te, f$best_th3)[, -1, drop = FALSE])
+         
+         colnames(Phi1_te) <- paste0("T1_leaf", 2:(ncol(Phi1_te) + 1))
+         colnames(Phi2_te) <- paste0("T2_leaf", 2:(ncol(Phi2_te) + 1))
+         colnames(Phi3_te) <- paste0("T3_leaf", 2:(ncol(Phi3_te) + 1))
+         
+         # Augment test dataframe with computed bases
+         test_augmented <- cbind(test, Phi1_te, Phi2_te, Phi3_te)
+         
+         # Safe prediction on test set using fixed effects only (re.form = ~0)
+         yhat_test_base <- predict(f$base, newdata = test_augmented, re.form = ~0)
+         yhat_test_soft <- predict(f$soft, newdata = test_augmented, re.form = ~0)
+         
+         mse_b[i] <- mean((test$Y - yhat_test_base)^2)
+         mse_s[i] <- mean((test$Y - yhat_test_soft)^2)
+         
+         anova_res <- anova(f$base, f$soft)
+         sig[i]    <- anova_res$`Pr(>Chisq)`[2] < 0.05
+     }
+     
+     data.frame(
+         Scenario      = sc,
+         Power         = mean(sig),
+         Test_MSE_Base = mean(mse_b),
+         Test_MSE_Soft = mean(mse_s),
+         Improvement   = mean(mse_b - mse_s)
+     )
+ })
