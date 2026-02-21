
# --- 加载依赖库 ---
library(ggplot2)
library(reshape2)
library(deSolve)
library(Matrix)
library(pracma)
library(patchwork)
library(ADSIHT)
library(MASS)
library(orthopolynom)

# --- 辅助函数：计算指标 ---
calculate_metrics <- function(predicted, true) {
  predicted <- as.integer(as.vector(predicted))
  true      <- as.integer(as.vector(true))
  
  TP <- sum(predicted == 1 & true == 1)
  FP <- sum(predicted == 1 & true == 0)
  TN <- sum(predicted == 0 & true == 0)
  FN <- sum(predicted == 0 & true == 1)
  
  TPR <- ifelse((TP + FN) == 0, 0, TP / (TP + FN))
  FPR <- ifelse((FP + TN) == 0, 0, FP / (FP + TN))
  
  denominator <- sqrt(as.numeric(TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
  MCC <- ifelse(denominator == 0, 0, (TP * TN - FP * FN) / denominator)
  
  list(TPR = TPR, FPR = FPR, MCC = MCC)
}

# --- 辅助函数：多项式基 ---
poly_basis_1d <- function(x, name = NULL, degree = 3) {
  n <- length(x)
  X <- matrix(NA, n, degree)
  for (d in 1:degree) {
    X[, d] <- x^d
  }
  base_name <- if (!is.null(name)) name else "x"
  colnames(X) <- paste0(base_name, "_", 1:degree)
  return(X)
}

# --- 模拟数据生成 ---
sim_data <- function(n = 100, p = 10, snr = 10, interaction = FALSE, sparsity = 0.2, seed = 123){
  
  set.seed(seed)
  # 稍微增加时间长度，让振荡展示得更充分
  times <- seq(0, 10, length = n)
  
  # --- 1. 修改基函数：加入线性项 x ---
  # 振荡通常需要线性项驱动 (dx/dt = ax...)
  # 现在的基是: [x, x^2, x^3]
  psi <- function(x) { c(x, x^2, x^3) }
  basisf = function(x){ cbind(x, x^2, x^3) }
  n_order = 3 # 阶数变为3
  
  # --- 2. 构建系数矩阵 (thetas) ---
  # 维度: (p * n_order) 行, p 列
  thetas <- matrix(0, nrow = p * n_order, ncol = p)
  
  # 为了产生耦合振荡，我们先生成一个随机的“交互骨架”
  # 使用偏斜对称矩阵 (A_ij = -A_ji) 有助于产生旋转/振荡
  interaction_skeleton <- matrix(rnorm(p*p, 0, 1), p, p)
  # 增加反对称成分权重，促进旋转
  interaction_skeleton <- (interaction_skeleton - t(interaction_skeleton)) * 0.8 + interaction_skeleton * 0.2
  
  for (j in 1:p) {
    # --- A. 自身项设置 ---
    
    # [Row 1] 线性项 (Linear, x): 控制局部生长/衰减
    # 设置为正值可以推动离开原点，引发振荡
    lin_idx <- (j - 1) * n_order + 1
    thetas[lin_idx, j] <- runif(1, 0.1, 0.5) 
    
    # [Row 2] 平方项 (Quadratic, x^2): 引入不对称性/不规律性
    sq_idx  <- (j - 1) * n_order + 2
    thetas[sq_idx, j] <- runif(1, -0.3, 0.3)
    
    # [Row 3] 立方项 (Cubic, x^3): 强负反馈，作为"盖子"防止发散
    # 这是产生稳定极限环的关键
    cube_idx <- (j - 1) * n_order + 3
    thetas[cube_idx, j] <- -runif(1, 1.0, 2.0) 
    
    # --- B. 互作项设置 (Cross-regulation) ---
    # 使用上面生成的 interaction_skeleton 决定连接
    others <- setdiff(1:p, j)
    
    # 随机选择连接对象，但参考 sparsity
    # 为了保证不规律的波动，我们允许更多的弱连接
    n_parents <- rbinom(1, length(others), sparsity)
    
    if (n_parents > 0) {
      parents <- sample(others, n_parents)
      for (parent in parents) {
        # 决定是线性耦合还是非线性耦合
        # 线性耦合 (Term 1) 最容易传递振荡
        # 非线性耦合 (Term 2, 3) 增加不规律性
        
        # 混合策略：主要使用线性耦合传递波动
        coupling_strength <- interaction_skeleton[parent, j] # 使用骨架权重
        
        # 线性项 (x_parent -> dx_j/dt)
        row_lin <- (parent - 1) * n_order + 1
        thetas[row_lin, j] <- coupling_strength * runif(1, 0.5, 1.0)
        
        # 偶尔添加非线性干扰 (x_parent^2 -> dx_j/dt)
        if(runif(1) < 0.3) {
          row_sq <- (parent - 1) * n_order + 2
          thetas[row_sq, j] <- runif(1, -0.2, 0.2)
        }
      }
    }
  }
  
  # --- 3. 交互项 (Synergistic Interactions) ---
  # x_i * x_j 项
  combinations <- combn(p, 2)
  n_pairs <- ncol(combinations)
  thetas_ij <- matrix(0, nrow = n_pairs, ncol = p)
  name_ij <- apply(combinations, 2, function(col) paste0("x", col[1], "*x", col[2]))
  
  if (interaction) {
    n_targets <- max(1, round(p * 0.3))
    targets <- sample(1:p, n_targets)
    for (target in targets) {
      n_int_terms <- sample(1:2, 1)
      pair_indices <- sample(1:n_pairs, n_int_terms)
      for (pair_idx in pair_indices) {
        # 交互项系数
        thetas_ij[pair_idx, target] <- runif(1, 0.5, 1.5) * sample(c(1, -1), 1)
      }
    }
  }
  
  # --- 4. ODE 定义 ---
  mod1 <- function(Time, State, Pars) {
    with(as.list(c(State, Pars)), {
      thetas = Pars$thetas
      thetas_ij = Pars$thetas_ij
      
      # StateBasis 现在包含 [x, x^2, x^3]
      StateBasis <- as.vector(sapply(State, psi)) 
      dX_main <- StateBasis %*% thetas
      
      dX_int <- rep(0, length(State))
      if (sum(abs(thetas_ij)) > 0) {
        pairwise_prod <- combn(State, 2, FUN = prod)
        dX_int <- pairwise_prod %*% thetas_ij
      }
      
      dX <- dX_main + dX_int
      
      # 软限制防止溢出 (Soft Clipping): 如果速度过快，进行抑制
      # 这在模拟高度非线性系统时很有用
      max_deriv <- 10
      dX <- pmax(pmin(dX, max_deriv), -max_deriv)
      
      if (any(is.infinite(dX)) || any(is.na(dX))) warning("Derivative infinite or NA")
      return(list(dX))
    })
  }
  
  # 初始状态：范围稍大一点，且随机化，让不同变量处于不同相位
  State <- runif(p, -1, 1)
  parms <- list(thetas = thetas, thetas_ij = thetas_ij)
  
  out <- tryCatch({
    # 使用 lsoda 求解，它能很好地处理刚性和振荡系统
    ode(y = State, times = times, func = mod1, parms = parms, method = "lsoda")
  }, error = function(e) { return(NULL) })
  
  if (is.null(out)) return(NULL)
  
  out_mat <- out[, -1]
  colnames(out_mat) <- paste0("x", 1:p)
  
  # 添加噪声
  signal_var <- apply(out_mat, 2, var)
  noise_sd <- sqrt(signal_var / snr)
  noise_sd[noise_sd < 1e-6] <- 1e-6
  
  Y <- out_mat + matrix(rnorm(n * p, sd = rep(noise_sd, each = n)), nrow = n, ncol = p)
  colnames(Y) <- paste0("x", 1:p)
  
  # --- 构建真实邻接矩阵 ---
  adj_main <- matrix(0, p, p)
  for (target in 1:p) {
    col_coeffs <- thetas[, target]
    for (source in 1:p) {
      idx_start <- (source - 1) * n_order + 1
      idx_end <- source * n_order
      # 只要该源变量的任意阶项系数不为0
      if (sum(abs(col_coeffs[idx_start:idx_end])) > 1e-6) adj_main[target, source] <- 1
    }
  }
  rownames(adj_main) <- colnames(adj_main) <- paste0("x", 1:p)
  
  return(list(p = p, times = times, Y = Y, adj_main = adj_main))
}
# --- 核心算法 ---
MTODE <- function(Y, times, M = 5, smooth = "bs") {
  if( is.null(colnames(Y))){
    colnames(Y) = paste0("x",1:ncol(Y))
  }
  
  
  n <- length(times)
  p <- ncol(Y)
  
  original_min <- min(times)
  original_range <- max(times) - min(times)
  # 归一化时间
  times_norm <- (times - min(times)) / (max(times) - min(times))
  times_new = seq(min(times_norm),max(times_norm),length = 50)
  times_restored <- times_new * original_range + original_min
  
  # 1. 预平滑 (Smoothing Splines)
  if (smooth == "bs") {
    fit_spline <- lapply(1:p, function(xi) smooth.spline(times_norm, as.numeric(Y[, xi])))
    x_smooth <- sapply(1:p, function(xi) predict(fit_spline[[xi]], x = times_norm)$y)
    colnames(x_smooth) <- colnames(Y)
    
    x_smooth2 <- sapply(1:p, function(xi) predict(fit_spline[[xi]], x = times_new)$y)
    colnames(x_smooth2) <- colnames(Y)
    
  } else if(smooth == "power_equation"){
    
    # 使用 sapply 遍历每一列进行非线性拟合
    x_smooth <- sapply(1:p, function(xi) {
      y_val <- as.numeric(Y[, xi])
      
      # 加上微小偏移量防止 0 的负指数导致无穷大
      x_val <- times_norm + 1e-6 
      
      # 使用 tryCatch 处理拟合不收敛的情况
      tryCatch({
        # 1. 初始值估计 (Initial Guesses)
        # 假设 y = a * x^b
        # a 估计为范围，b 初始设为 1 (线性)
        c_start <- min(y_val)
        a_start <- max(y_val) - min(y_val)
        if(a_start == 0) a_start <- 0.1 # 防止常数数列导致错误
        
        # 2. 非线性最小二乘拟合 (NLS)
        fit_nls <- nls(y_val ~ a * I(x_val^b), 
                       start = list(a = a_start, b = 1),
                       control = nls.control(maxiter = 100, warnOnly = TRUE))
        
        # 3. 返回预测值
        predict(fit_nls)
        
      }, error = function(e) {
        # 4. 回退机制 (Fallback)
        # 如果数据不符合幂律分布（例如是波动的），NLS 会失败。
        # 此时回退到局部加权回归 (LOESS) 以保证程序不中断。
        warning(paste("Var", colnames(Y)[xi], ": Power fit failed, using Loess instead."))
        fitted(loess(y_val ~ times_norm, span = 0.75))
      })
    })
    # 保持列名一致
    colnames(x_smooth) <- colnames(Y)
  }
  
  
  
  # 2. 构建基函数并积分
  # 构造多项式基
  phi_list <- lapply(1:p, function(xi) {
    poly_basis_1d(x_smooth[, xi], name = colnames(Y)[xi], degree = M)
  })
  phi <- Reduce(cbind, phi_list)
  
  # 对基进行积分 (Trapezoidal integration)
  phi_int <- apply(phi, 2, function(col) cumtrapz(times_norm, col))
  
  # --- 定义组构造函数 ---
  construct_group <- function(j, phi_int_data) {
    s0 <- p * (j - 1) + 1
    group <- rep(s0:(s0 + p - 1), each = M)
    group0 <- min(group)
    group <- sort(c(group0, group))
    group_levels <- unique(group)
    
    phi1_int_local <- cbind(times_norm, phi_int_data)
    group_idx_list <- split(seq_len(ncol(phi1_int_local)), group)
    
    qr_list <- lapply(group_levels, function(g) {
      cols <- group_idx_list[[as.character(g)]]
      Xg <- phi1_int_local[, cols, drop = FALSE]
      mXg <- colMeans(Xg)
      Xg_centered <- sweep(Xg, 2, mXg, FUN = "-")
      
      # 若秩不足则加微小噪声
      if (qr(Xg_centered)$rank < ncol(Xg_centered)) {
        Xg_centered <- Xg_centered + matrix(rnorm(length(Xg_centered), sd = 1e-12),
                                            nrow = nrow(Xg_centered), ncol = ncol(Xg_centered))
      }
      qr_g <- qr(Xg_centered)
      list(Q = qr.Q(qr_g, complete = FALSE), R = qr.R(qr_g), mXg = mXg, src_cols = cols)
    })
    
    Q_all <- do.call(cbind, lapply(qr_list, `[[`, "Q"))
    
    list(
      group = group,
      group_idx_list = group_idx_list,
      phi_int = phi_int_data,
      Q_all = Q_all,
      R_list = lapply(qr_list, `[[`, "R"),
      mXg_list = lapply(qr_list, `[[`, "mXg"),
      src_cols_list = lapply(qr_list, `[[`, "src_cols")
    )
  }
  
  # 3. 构造大设计矩阵 (Block Diagonal Q)
  Q_info <- lapply(1:p, construct_group, phi_int_data = phi_int)
  Q_blocks <- lapply(Q_info, `[[`, "Q_all")
  X_design <- as.matrix(bdiag(Q_blocks))
  
  # 4. 数据标准化与回归
  y_scaled <- scale(Y)
  Y_all <- matrix(as.vector(y_scaled), ncol = 1)
  
  groups_vec <- unlist(lapply(Q_info, `[[`, "group"))
  
  # ADSIHT 回归
  # fit <- ADSIHT(X_design, Y_all, group = groups_vec)
  
  fit <- ADSIHT(X_design, Y_all, group = groups_vec)
  
  
  best_idx <- which.min(fit$ic)
  beta_std_Q_vec <- fit$beta[, best_idx]
  
  #fit <- cv.sparsegl(X_design, Y_all, group = groups_vec) #sparsegl
  #fit <- cv.grpreg(X_design, Y_all, group = groups_vec)   #grpreg
  #beta_std_Q_vec <- as.vector(coef(fit, s = fit$lambda.1se)[-1])
  
  # 5. 恢复原始 Beta
  recover_beta <- function(beta_std_Q_vec, Q_info, y_s) {
    cols_per_j <- sapply(Q_info, function(z) ncol(z$Q_all))
    cum_cols <- c(0, cumsum(cols_per_j))
    sY <- attr(y_s, "scaled:scale")
    mY <- attr(y_s, "scaled:center")
    
    res_list <- lapply(seq_along(Q_info), function(j) {
      idx_range <- (cum_cols[j] + 1):cum_cols[j + 1]
      bQj_all <- beta_std_Q_vec[idx_range]
      
      info <- Q_info[[j]]
      rcols_vec <- sapply(info$R_list, ncol)
      bQg_list <- split(bQj_all, rep(seq_along(rcols_vec), rcols_vec))
      
      beta_Xj_full <- numeric(ncol(info$phi_int))
      mX_full <- numeric(ncol(info$phi_int))
      
      Map(function(Rg, bQg, src_cols, mXg) {
        # 注意: backsolve 结果写入父环境变量
        beta_Xg <- backsolve(Rg, bQg)
        beta_Xj_full[src_cols] <<- beta_Xg
        mX_full[src_cols] <<- mXg
      }, info$R_list, bQg_list, info$src_cols_list, info$mXg_list)
      
      beta_orig_j <- sY[j] * beta_Xj_full
      intercept_orig_j <- mY[j] - sY[j] * sum(mX_full * beta_Xj_full)
      list(beta = beta_orig_j, intercept = intercept_orig_j)
    })
    
    list(
      B_est = lapply(res_list, `[[`, "beta"),
      intercept = sapply(res_list, `[[`, "intercept")
    )
  }
  
  beta_recovered <- recover_beta(beta_std_Q_vec, Q_info, y_scaled)
  beta_orig <- Reduce(cbind, beta_recovered$B_est)
  
  beta_hat <- beta_orig[-1, ]
  beta_all <- rbind(beta_recovered$intercept, beta_orig)
  
  
  
  # 6. 计算估计值与绘图
  #new_phi_all =  cbind(1, as.matrix(bdiag(cbind(times_norm, phi_int))))
  phi_all <- cbind(1, as.matrix(bdiag(cbind(times_norm, phi_int))))
  y_est <- phi_all %*% beta_all
  
  y_est_df <- reshape2::melt(data.frame(times = times, y_est), id.vars = 'times')
  y_obs_df <- reshape2::melt(data.frame(times = times, Y), id.vars = 'times')
  
  pp0 <- ggplot() + 
    geom_point(data = y_obs_df, aes(times, value, group = variable)) +
    geom_line(data = y_est_df, aes(times, value, group = variable), color = 'red') +
    facet_wrap(~variable) + 
    theme_bw()
  
  
  
  
  
  # 找出固定为0的参数索引
  par_init <- c(beta_orig)        # 初始参数向量
  par_mat <- matrix(par_init, ncol = ncol(Y))
  fixed_idx <- which(par_init == 0)
  free_idx  <- setdiff(seq_along(par_init), fixed_idx)
  
  # ridge目标函数
  ridge_loss <- function(free_par, y, x, par_init, free_idx, lambda = 1e-3) {
    # 保持0参数固定
    par_full <- par_init
    par_full[free_idx] <- free_par
    
    par_mat <- matrix(par_full, ncol = ncol(y))
    est <- x %*% par_mat
    err <- sum((y - est)^2)
    
    penalty <- lambda * sum(free_par^2)   # ridge惩罚
    return(err + penalty)
  }
  
  # 初始值（仅优化非零参数）
  free_init <- par_init[free_idx]
  
  # 优化
  opt_res <- optim(
    par = free_init,
    fn = ridge_loss,
    y = sweep(Y, 2, beta_recovered$intercept, "-"),
    x = as.matrix(bdiag(cbind(times_norm, phi_int))),
    par_init = par_init,
    free_idx = free_idx,
    method = "L-BFGS"#,
    #control = list(trace = "T")
  )
  
  # 得到最终参数（原本为0的保持0）
  par_final <- par_init
  par_final[free_idx] <- opt_res$par
  par_final_mat <- matrix(par_final, ncol = ncol(Y))
  beta_all = rbind(beta_recovered$intercept,par_final_mat)
  
  
  
  
  
  
  
  phi_list2 <- lapply(1:p, function(xi) {
    poly_basis_1d(x_smooth2[, xi], name = colnames(Y)[xi], degree = M)
  })
  phi2 <- Reduce(cbind, phi_list2)
  
  # 对基进行积分 (Trapezoidal integration)
  phi_int2 <- apply(phi2, 2, function(col) cumtrapz(times_new, col))
  phi_all2 <- cbind(1, as.matrix(bdiag(cbind(times_new, phi_int2))))
  
  
  y_est2 <- phi_all2 %*% beta_all
  
  y_est_df2 <- reshape2::melt(data.frame(times = times_restored, y_est2), id.vars = 'times')
  y_obs_df <- reshape2::melt(data.frame(times = times, Y), id.vars = 'times')
  
  pp1 <- ggplot() + 
    geom_point(data = y_obs_df, aes(times, value, group = variable)) +
    geom_line(data = y_est_df2, aes(times, value, group = variable), color = 'red') +
    facet_wrap(~variable) + 
    theme_bw()
  
  
  # 7. 计算组贡献 (f_group_est)
  f_group <- function(jj) {
    
    betam <- beta_all[-c(1:2), jj]
    est <- matrix(0, nrow = length(times_new), ncol = p)
    
    group_idx_list <- split(seq_len(ncol(phi2)), rep(1:p, each = M))
    
    for (g in seq_len(p)) {
      est[, g] <- phi_int2[, group_idx_list[[g]], drop = FALSE] %*% betam[group_idx_list[[g]]]
    }
    
    # 加上截距项和时间线性项
    est[, jj] <- est[, jj] + beta_all[2, jj] * times_new +  beta_all[1, jj]
    
    colnames(est) <- colnames(Y)
    return(est)
  }
  
  f_group_est <- lapply(1:p, f_group)
  
  adj_main_est <- matrix(0, nrow = p, ncol = p)
  for (i in 1:p) {
    # 只要该组估计值总和不全为0，即认为存在边
    adj_main_est[i, which(colSums(f_group_est[[i]]) != 0)] <- 1
  }
  rownames(adj_main_est) = colnames(Y)
  colnames(adj_main_est) = colnames(Y)
  
  
  
  return(list(beta_all = beta_all,
              times = times,
              f_group_est = f_group_est, 
              adj_est = adj_main_est,
              fig0 = pp0,
              fig = pp1))
}

# --- 主循环：模拟与评估 ---
n_sim <- 2
results_main <- matrix(NA, nrow = n_sim, ncol = 3) 


total_start_time <- Sys.time() # 记录总开始时间
for (sim_i in 1:n_sim) {
  cat("Simulation:", sim_i, "\n")
  
  dat <- sim_data(p = 100, seed = 100 + sim_i, snr = 30)
  
  # 拟合模型
  res <- MTODE(Y = dat$Y, times = dat$times, M = 3)
  
  metrics_main <- calculate_metrics(res$adj_est, dat$adj_main)
  results_main[sim_i, ] <- c(metrics_main$TPR, metrics_main$FPR, metrics_main$MCC)
}
total_end_time <- Sys.time() # 记录总结束时间
total_duration <- total_end_time - total_start_time

summary_main <- data.frame(
  Mean = apply(results_main, 2, mean),
  SD = apply(results_main, 2, sd),
  row.names = c("TPR", "FPR", "MCC")
)

cat("=== 结果汇总 ===\n")
print(summary_main)
print(total_duration)
print(res$fig0)
