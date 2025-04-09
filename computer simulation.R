rm(list = ls())
library(Matrix)
library(deSolve)
library(orthopolynom)
library(ggplot2)
library(ADSIHT)
library(reshape2)
require(deSolve)
library(patchwork)
library(glmnet)
library(sparsegl)
set.seed(1)
p <- 6  
n_order = 1
sd = 0.01
m=50
rep=10

sim_data <- function(p=p,m=m,n=n,sd=sd){
  ks <- seq(from = 1, to = p/2, length.out = p/2)
  ks <- ks * pi * 1.7
  thetas <- matrix(0, p, p)
  for(j in 1:(p/2)){
    thetas[2*j-1, 2*j] <- ks[j]
    thetas[2*j, 2*j-1] <- -ks[j]
  }
  diag(thetas) <- rnorm(p)
  intercepts <- rep(0.2, p)
  thetas <- cbind(intercepts, thetas)
  
  upp <- 1   # time range
  tgrid <- seq(from = upp/m, to = upp, length = m)
  mod <- function(Time, State, Pars) {
    with(as.list(c(State, Pars)), {
      dxdt <- A %*% State + r
      return(list(dxdt))
    })
  }
  inis <- rnorm(p/2)
  initialvalue <- c(rbind(sin(inis), cos(inis)))
  A <- thetas[, -1]
  r <- thetas[, 1]
  State <- initialvalue
  
  # Solve the ODE system (each column of out corresponds to a gene/state)
  out <- ode(y = State, times = tgrid, 
             func = mod, parms = list(A = A, r = r), 
             method = "ode45")[, -1]
  
  matplot(tgrid, out, type = "l", lty = 1, 
          xlab = "Time", ylab = "State variables", 
          main = "ODE Output", col = 1:p)
  legend("topright", legend = paste0("gene", 1:p), col = 1:p, lty = 1, cex = 0.6)
  
  #### Data generation ####
  data = out[seq(1,nrow(out), length.out = n),] + matrix(rnorm(n*p, sd = sd),n,p,byrow = T)
  times = tgrid[seq(1,nrow(out), length.out = n)]
  
  return(list(thetas = thetas,
              tgrid = tgrid, 
              times = times,
              out = out, 
              data = data))
}
test_x <- function(d=d,n=n){
  idx = seq(3,(p*n_order+2))
  idx_list = rep(list(split(idx, cut(seq_along(idx), breaks = p, labels = FALSE))), times = 6)
  idx_list <- lapply(seq_along(idx_list), function(i) {
    # 将 1 和 2 插入到 idx_list[[i]][i] 的前面
    idx_list[[i]][[i]] = c(1, 2, idx_list[[i]][[i]])
    return(idx_list[[i]])
  })
  
  thetas = d$thetas
  tgrid = d$tgrid
  times = d$times
  out = d$out
  data = d$data
  
  # ture effect
  d_effect_list <- lapply(1:p, function(j) {
    d_effect <- sweep(cbind(1,out), 2, thetas[j,], "*")
    d_effect[,j+1] <- d_effect[,j+1]+d_effect[,1]
    d_effect <- d_effect[,-1]
  })
  
  d_effect_fs_list <- lapply(d_effect_list, function(mat) {
    apply(mat, 2, function(col) splinefun(x = tgrid, y = col))
  })
  
  mod <- function (Time, State, Pars, basis) {
    with(as.list(c(State, Pars)), {
      dy = basis(Time)
      return(list(dy))
    })
  }
  Pars <- NULL
  State <- matrix(0,p,p)
  diag(State) <- out[1,]
  
  effect_list <- lapply(seq_along(d_effect_fs_list), function(i) {
    mat <- d_effect_fs_list[[i]]
    sapply(1:length(mat),function(xi) 
      ode(func = mod, y = State[i,xi], parms = Pars, 
          basis = mat[[xi]], times = tgrid)[,2])
  })
  
  #截距加到独立效应中
  trans_out <- function(out,j){
    id = paste0("gene",j)
    
    out = as.data.frame(out)
    colnames(out) = c('time',"est",paste0("gene",1:p))
    out = melt(out,id.vars = 'time')
    out$group = 'Dep'
    out$group[out$variable == id] = 'Ind'
    out$group[out$variable == 'est'] = 'est'
    out
  }
  trans_effect_list <- lapply(seq_along(effect_list), function(i)
    effect = trans_out(cbind(tgrid,rowSums(effect_list[[i]]),effect_list[[i]]),i))
  effect_all <- do.call(rbind, trans_effect_list)
  effect_all$equations <- rep(paste0("equations", 1:p), each = m*(p+1))
  
  p0 = ggplot(effect_all) + 
    geom_line(mapping = aes(x=time,y = value,group = variable,color = group))+
    scale_color_manual(values = c(est = 'blue',
                                  Ind = 'red',
                                  Dep = 'green'))+
    theme_minimal() + facet_wrap(~equations)
  p0
  
  # 双稀疏回归
  ODEsolve <- function(data,times=times,n_order=n_order){
    
    p = ncol(data)
    #smooth data
    fit <- lapply(1:p, function (xi) 
      smooth.spline(times, as.numeric(data[,xi]), spar = 0.5))
    
    x_smooth <- sapply(1:p, function(xi)  predict(fit[[xi]], x = times)$y)
    
    group <- rep(1:(p*p), each=n_order+1)
    ind_Leg0 <- c(0:(p-1))*(p*(n_order+1))+1 + rep(0:(p-1)) * (n_order+1)
    Leg0 <- seq(1,length(group), by = n_order+1)
    dep_Leg0 <- setdiff(Leg0, ind_Leg0)
    group <- group[-dep_Leg0]
    
    get_LOP_M <- function(x, n_order,times, name = NULL) {
      # Generate Legendre basis (excluding the constant term)
      x <- as.numeric(x)
      LOP <- legendre.polynomials(n = n_order)
      leg <- polynomial.values(polynomials=LOP,x = scaleX(x, u = -1, v = 1))
      #integral over times
      fs = lapply(leg,splinefun,x = times)
      mod <- function (Time, State, Pars, basis) {
        with(as.list(c(State, Pars)), {
          dy = basis(Time)
          return(list(dy))
        })
      }
      Pars <- NULL
      State <- rep(0,length(leg))
      #State <- lapply(leg, '[[', 1)
      #State[1] <- 0
      leg <- sapply(1:length(fs),function(xi) 
        ode(func = mod, y = State[[xi]], parms = Pars, 
            basis = fs[[xi]], times = times)[,2])
      
      if (is.null(name)) {
        colnames(leg) <- paste0("leg", 0:(ncol(leg)-1))
      } else {
        colnames(leg) <- paste0(name, "__leg", 0:(ncol(leg)-1))
      }
      # Drop the constant term (first column)
      #leg[, -1]
      leg
    }
    x_basis <- Reduce(cbind, lapply(1:p, function(xi) {
      get_LOP_M(x = x_smooth[, xi], n_order,times)
    }))
    x_scaled <- scale(x_basis)
    X <- as.matrix(bdiag(rep(list(x_scaled), p)))
    X <- X[, -dep_Leg0]
    y_scaled <- scale(data)
    Y <- matrix(as.vector(y_scaled), ncol = 1)
    order0_positions <- sapply(as.numeric(names(table(group)[table(group) == max(table(group))])), 
                               function(num) which(group == num)[1])
    ## 双稀疏
    time_ds <- system.time(
      res_ds <- ADSIHT(x = X, y = Y, group = group, L = n_order, 
                       kappa = 0.99, ic.scale = 1, ic.coef = 1))
    beta_ds <- res_ds$beta[,which.min(res_ds[["ic"]])]
    beta0_scaled_ds <- res_ds$intercept[which.min(res_ds[["ic"]])]
    
    ## lasso
    time_lasso <- system.time({
      cv_lasso <- cv.glmnet(X, Y, alpha = 1, nfolds = 5) 
      beta_lasso <- coef(cv_lasso, s = "lambda.min")
    })
    beta_lasso <- as.vector(coef(lasso_final_model)[-1])
    beta0_scaled_lasso <- coef(lasso_final_model)[1]
    # matrix(beta_lasso,p+1,p)
    ## sglasso
    time_sglasso <- system.time({
      cv_fit <- cv.sparsegl(X, Y, group = group, nfolds = 5)
      beta_sglasso <- coef(cv_fit, s = "lambda.min")
    })
    beta_sglasso <- as.vector(beta_sglasso[-1])
    beta0_scaled_sglasso <- beta_sglasso[1]
    
    # turn beta
    get_B_est <- function(beta=beta,beta0_scaled=beta0_scaled) {
      beta_est <- matrix(beta[-order0_positions], nrow = p * n_order, ncol = p)
      beta_est <- rbind(beta[order0_positions],beta_est)
      
      sY <- attr(y_scaled, "scaled:scale")
      mY <- attr(y_scaled, "scaled:center")
      sX <- attr(x_scaled, "scaled:scale")
      sX <- c(sX[1],sX[!grepl("leg0", names(sX))])
      mX <- attr(x_scaled, "scaled:center")
      mX <- c(mX[1],mX[!grepl("leg0", names(mX))])
      
      BETA <- lapply(1:p, function(xi) {
        beta_orig <- beta_est[, xi] * (sY[xi] / sX)
        return(beta_orig)
      })
      BETA0 <- lapply(1:p, function(xi) {
        orig_int <- beta0_scaled * sY[xi] + mY[xi] - sum(BETA[[xi]] * mX)
        return(orig_int)
      })
      B_est = t(cbind(unlist(BETA0),t(Reduce(cbind,BETA))))
      B_est
    }
    B_est_ds <- get_B_est(beta_ds, beta0_scaled = beta0_scaled_ds)
    B_est_lasso <-get_B_est(beta_lasso, beta0_scaled = beta0_scaled_lasso)
    B_est_sglasso <-get_B_est(beta_sglasso, beta0_scaled = beta0_scaled_sglasso)
    
    x_basis <- cbind(x_basis[,1],x_basis[,!grepl("leg0", colnames(x_basis))])
    x_basis_est = cbind(1,x_basis)
    
    return(list(x_basis = x_basis_est,
                B_est_ds = B_est_ds,
                B_est_lasso = B_est_lasso,
                B_est_sglasso = B_est_sglasso,
                x_smooth = x_smooth,
                time_ds = time_ds,
                time_lasso = time_lasso,
                time_sglasso = time_sglasso
    )) 
  }
  res <- ODEsolve(data = data, times = times, n_order = n_order)
  
  B_est_ds = res$B_est_ds
  B_est_lasso = res$B_est_lasso
  B_est_sglasso = res$B_est_sglasso
  x_basis_est = res$x_basis
  
  get_effect_plot <- function(B_est=B_est, x_basis_est = x_basis_est, times = times){
    trans_effect_est_list <- lapply(1:p, function(j) {
      effect_est <- cbind(times, rowSums(sapply(1:p, function(xi) x_basis_est[,idx_list[[j]][[xi]]] %*% B_est[idx_list[[j]][[xi]], j])),
                          sapply(1:p, function(xi) x_basis_est[,idx_list[[j]][[xi]]] %*% B_est[idx_list[[j]][[xi]], j]))
      trans_out(effect_est, j)
    })
    effect_est_all <- do.call(rbind, trans_effect_est_list)
    effect_est_all$equations <- rep(paste0("equations", 1:p), each = n*(p+1))
    ori <- as.data.frame(cbind(times,data))  
    ori <- melt(ori,id.vars = 'times')
    ori$equations <- rep(paste0("equations", 1:p), each = n)
    
    p2 = ggplot() + geom_point(ori,mapping = aes(x=times, y = value),color = 'blue',size=0.1)+
      geom_line(effect_all, mapping = aes(x=time,y = value,group = variable,color = group))+
      geom_line(effect_est_all, mapping = aes(x=time,y = value,group = variable,color = group),linetype = 2)+
      scale_color_manual(values = c(est = 'blue',
                                    Ind = 'red',
                                    Dep = 'green'))+
      theme_minimal() + facet_wrap(~equations)
    p2
  }
  effect_plot_ds <- get_effect_plot(B_est=B_est_ds, x_basis_est=x_basis_est, times = times)
  effect_plot_lasso <- get_effect_plot(B_est=B_est_lasso, x_basis_est=x_basis_est, times = times)
  effect_plot_sglasso <- get_effect_plot(B_est=B_est_sglasso, x_basis_est=x_basis_est, times = times)
  
  return(list(effect_plot_ds = effect_plot_ds,
              effect_plot_lasso = effect_plot_lasso,
              effect_plot_sglasso = effect_plot_sglasso,
              B_est_ds = B_est_ds,
              B_est_lasso = B_est_lasso,
              B_est_sglasso = B_est_sglasso,
              x_basis_est = x_basis_est,
              time_ds = res$time_ds,
              time_lasso = res$time_lasso,
              time_sglasso = res$time_sglasso))
}
tpr_fpr <- function(estimated, true_values) {
  est_bin <- (estimated != 0) * 1
  true_bin <- (true_values != 0) * 1
  
  tp <- sum(est_bin == 1 & true_bin == 1)
  fn <- sum(est_bin == 0 & true_bin == 1)
  tpr <- tp / (tp + fn)
  
  fp <- sum(est_bin == 1 & true_bin == 0)
  tn <- sum(est_bin == 0 & true_bin == 0)
  fpr <- fp / (fp + tn)
  
  accuracy <- (tp + tn) / (tp + tn + fp + fn)
  
  mcc_numerator <- (tp * tn) - (fp * fn)
  mcc_denominator <- sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
  mcc <- ifelse(mcc_denominator == 0, 0, mcc_numerator / mcc_denominator)
  
  return(list(TPR = tpr, FPR = fpr, accuracy = accuracy, MCC = mcc))
}


n_values <- c(20, 30, 40, 50)
results <- list()
for (n_val in n_values) {
  # 创建一个存储每个n值下重复计算结果的列表
  results[[paste("n", n_val, sep = "_")]] <- list()
  
  for (i in 1:rep) {
    d <- sim_data(p = p, m = m, n = n_val, sd = sd)
    res <- test_x(d = d, n = n_val)
    results[[paste("n", n_val, sep = "_")]][[paste("repeat", i, sep = "_")]] <- res
  }
}
