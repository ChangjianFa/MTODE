rm(list = ls())
library(ADSIHT)
library(mvtnorm)
library(pbapply)
library(parallel)
library(deSolve)
library(orthopolynom)
library(glmnet)
library(ggplot2)
library(reshape2)
library(MASS)
library(patchwork)
library(idopNetwork)
library(Matrix)
library(tidyr)
library(stringr)
library(mgcv)#拟合
library(nlme)#拟合
library(polynom)#多项式积分
DSIHT_Cpp <- function(x, y, weight, ic_type, ic_scale, sequence, kappa, g_index, ic_coef, method, coef1, coef2, eta, max_iter,nor) {
  .Call('_ADSIHT_DSIHT_Cpp', PACKAGE = 'ADSIHT', x, y, weight, ic_type, ic_scale, sequence, kappa, g_index, ic_coef, method, coef1, coef2, eta, max_iter,nor)
}
ADSIHT <- function(x, y, group,
                   s0,  #####given L, s0 =  max(table(group))^(seq(1, L-1, length.out = L)/(L-1)), given L, donot need s0
                   kappa = 0.8,########0.9-1
                   ic.type = c("dsic"),
                   ic.scale = 0.1,###***** constant
                   ic.coef = 0.1,####越小边的数量越多*****constant
                   L = n_order+1,########
                   weight = rep(1, nrow(x)),
                   coef1 = 1,
                   coef2 = 1,
                   eta = 0.8,####越大越好0.5-1,method 选ols则不起作用
                   max_iter = 50,
                   method = "ols")
{
  if(missing(group)) group <- 1:ncol(x)
  p <- ncol(x)
  n <- nrow(x)
  N <- length(unique(group))
  if (length(group)!= ncol(x)) stop("The length of group should be the same with ncol(x)")
  if(!is.matrix(x)) x <- as.matrix(x)
  vn <- colnames(x)
  orderGi <- order(group)
  x <- x[, orderGi]
  vn <- vn[orderGi]
  group <- group[orderGi]
  gi <- unique(group)
  index <- match(gi, group)-1
  if (missing(s0)) {
    s0 <- max(table(group))^(seq(1, L-1, length.out = L)/(L-1))
  }
  ic.type <- match.arg(ic.type)
  ic_type <- switch(ic.type,
                    "loss" = 0,
                    "dsic" = 1
  )
  if (method == "ols") {
    method = TRUE
  } else {
    method = FALSE
  }
  res <- DSIHT_Cpp(x, y, weight = weight, sequence = s0, ic_type = ic_type, ic_scale = ic.scale, kappa = kappa, g_index = index,
                   ic_coef = ic.coef, coef1 = coef1, coef2 = coef2, eta = eta, max_iter = max_iter, method = method,nor=T)
  return(res)
}

# 参数设置
set.seed(123)
n_vars <- 15  # 变量数量
n_order <- 4
time_points <- seq(0, 10, length.out = 100)  # 时间点
interaction_prob <- 0.2  # 稀疏互作的概率

# 随机生成稀疏的互作矩阵
interaction_matrix <- matrix(rbinom(n_vars+n_order*n_vars^2, 1, interaction_prob), n_vars, n_order*n_vars+1)
for (n in 1:n_vars) {
  if(all(interaction_matrix[n,((n-1)*n_order+1):(n*n_order+1)] == 0)){
    interaction_matrix[n,((n-1)*n_order+1):(n*n_order+1)][sample(1:9,2)] <- 1
  }
}

# 系统的随机参数
alpha <- matrix(runif(n_vars * (n_order+1), -0.1, 0.1), nrow = n_vars, ncol = n_order+1)  # 每个变量的增长速率
beta <- matrix(runif(n_vars * (n_vars-1) * n_order, -0.05, 0.05), nrow = n_vars, ncol = (n_vars-1) * n_order)  # 互作强度

get_LOP_M <- function(x, n_order,times = scaled_t,  name=NULL, scale=T){
  # Create basis matrix
  require(orthopolynom)
  
  LOP = legendre.polynomials(n = n_order)
  
  if (scale) {
    x=scaleX(x, u=-1, v=1)
  }
  
  leg <- polynomial.values(polynomials=LOP,x = x)
 
  if ( is.null(name) ) {
    leg <- as.matrix(as.data.frame(leg))
    colnames(leg) <- paste0("leg",0:(ncol(leg)-1))
  } else {
    colnames(leg) <- paste0(name,"__leg",0:(ncol(leg)-1))
  }
  #leg = leg[,-1] #remove zero order
  leg
}
s_X <- do.call(cbind, lapply(1:ncol(A), function(edge) {
  get_LOP_M( x=A[,edge] , n_order )
}))
# 定义微分方程组
interaction_model <- function(t, state, parameters) {
  n <- length(state)
  dx <- numeric(n)
  
  for (i in 1:n) {
    # 计算与变量i相关的互作项
    interaction_terms <- sum(interaction_matrix[i, ] * beta[i,] * state )
    dx[i] <- alpha[i] * state[i] + interaction_terms
  }

  list(dx)
}

# 初始状态
initial_state <- runif(n_vars, 0.5, 2)

# 求解微分方程
simulation <- ode(
  y = initial_state,
  times = time_points,
  func = interaction_model,
  parms = NULL
)

# 转换为数据框
simulation_data <- as.data.frame(simulation)

# 添加噪声
noise <- matrix(rnorm(100 * 15, mean = 0, sd = sqrt(0.001)), nrow = 100, ncol = 15)
simulation_data[,2:16] <- simulation_data[,2:16] + noise

# 可视化结果
library(ggplot2)
simulation_long <- reshape2::melt(simulation_data, id.vars = "time", variable.name = "Variable", value.name = "Value")
ggplot(simulation_long, aes(x = time, y = Value, color = Variable)) +
  geom_line() +
  labs(title = "Dynamics of 15 Interacting Variables", x = "Time", y = "Value") +
  theme_minimal()


data_n <- simulation_data[,-1]
times <- time_points
n_data_n <- scale(data_n)
scaled_t <- scale(times)
center <- attr(n_data_n, "scaled:center")
scale <- attr(n_data_n, "scaled:scale")

gam_ <- function(Y, t) {
  p <- ncol(Y)
  X_hat <- matrix(0, nrow = nrow(Y), ncol = p)
  for (j in 1:p) {
    fit <- gam(Y[, j] ~ s(t, bs = "tp", k = min(10, length(unique(t))) ))  # 使用
    X_hat[, j] <- predict(fit, newdata = data.frame(t = t))
  }
  return(X_hat)
}
X_hat <- gam_(Y=n_data_n, t=as.numeric(scaled_t))
plot_variable <- function(index, t, Y, X_hat) {
  data <- data.frame(
    Time = t,
    Observed = Y[, index],
    Predicted = X_hat[, index]
  )
  ggplot(data, aes(x = Time)) +
    geom_point(aes(y = Observed, color = "Observed"), size = 1) +
    geom_line(aes(y = Predicted, color = "Predicted"), linetype = "dashed", linewidth = 1) +
    labs(title = colnames(X_hat)[index], y = "Value") +
    theme_minimal() +
    scale_color_manual(values = c("Observed" = "blue", "Predicted" = "red")) +
    theme(legend.position = "none")
}
X_hat <- n_data_n
plots <- lapply(1:ncol(X_hat), function(i) plot_variable(i, t = as.numeric(scaled_t), Y = n_data_n, X_hat = X_hat))
big_plot <- wrap_plots(plots, ncol = 4) 
# ggsave("D:/data/computer simulation/variables.png", big_plot, width = 18, height = 12)  


A <- X_hat
n <- nrow(A)
t <- ncol(A)
n_order=9
group <- rep(1:(t*t), each=n_order)
ind_Leg0 <- c(0:(t-1))*(t*n_order)+1 + rep(0:(t-1)) * n_order
group <- c(group, ind_Leg0)
group <- sort(group)

get_LOP_M <- function(x, n_order,times = scaled_t,  name=NULL, scale=F){
  # Create basis matrix
  require(orthopolynom)
  
  x = as.numeric(x)
  LOP = legendre.polynomials(n = n_order)
  
  if (scale) {
    x=scaleX(x, u=-1, v=1)
  }
  
  leg <- polynomial.values(polynomials=LOP,x = x)
  
  if( length(times) > 0 ) {
    #integral over times to get PHI
    require(deSolve)
    
    fs = lapply(leg,splinefun,x = times)
    
    mod <- function (Time, State, Pars, basis) {
      with(as.list(c(State, Pars)), {
        dy = basis(Time)
        return(list(dy))
      })
    }
    Pars <- NULL
    State <- lapply(leg, '[[', 1)
    
    leg <- sapply(1:length(fs),function(xi) 
      ode(func = mod, y = State[[xi]], parms = Pars, 
          basis = fs[[xi]], times = times)[,2])
    
  }
  
  if ( is.null(name) ) {
    leg <- as.matrix(as.data.frame(leg))
    colnames(leg) <- paste0("leg",0:(ncol(leg)-1))
  } else {
    colnames(leg) <- paste0(name,"__leg",0:(ncol(leg)-1))
  }
  #leg = leg[,-1] #remove zero order
  leg
}
s_X <- do.call(cbind, lapply(1:ncol(A), function(edge) {
  get_LOP_M( x=A[,edge] , n_order )
}))

X <- bdiag(rep(list(s_X), t))
#间隔为n_order+1
ind_Leg0 <- c(0:(t-1))*(t*(n_order+1))+1 + rep(0:(t-1)) * (n_order+1)
Leg0 <- seq(1,ncol(X), by = n_order+1)
dep_Leg0 <- setdiff(Leg0, ind_Leg0)
X <- X[,-dep_Leg0]
Y <- matrix(as.vector(n_data_n), nrow = n*t, ncol = 1)
res <- ADSIHT( x=X , y=Y, group = group)



############plot
n_beta=which.min(res[["ic"]]) #n_beta=8
beta <- res$beta[,n_beta]
beta <- matrix(beta, nrow = length(group), ncol = 1)
y_est <- X %*% beta + res$intercept[n_beta]

beta <- matrix(beta, nrow = t*n_order+1, ncol = t)
y_est <- matrix(y_est, nrow = n, ncol = t)
Y <- matrix(Y, nrow = n, ncol = t)
colnames(beta) <- colnames(n_data_n)
colnames(y_est) <- colnames(n_data_n)
colnames(Y) = colnames(n_data_n)
rownames(y_est) <- rownames(n_data_n)
rownames(Y) = rownames(n_data_n)


########分解曲线
num_groups <- ncol(s_X) / (n_order+1)
result_list <- list() 

for (j in 1:num_groups) {
  group_result <- list()
  
  for (i in 1:num_groups) {
    if (i == j) {
      cols <- ((i - 1) * (n_order+1) + 1):(i * (n_order+1)) 
      rows <- ((i-1)*n_order+1):((i-1)*n_order+1+n_order)
      group_result[[i]] <- s_X[, cols] %*% beta[rows, j]
    }else{
      cols <- ((i - 1) * (n_order+1) + 1):(i * (n_order+1))
      if (i < j) {
        rows <- ((i - 1) * (n_order) + 1):(i * (n_order)) 
      } else {
        rows <- ((i - 1) * (n_order) + 2):(i * (n_order)+1) 
      }
      group_result[[i]] <- s_X[, cols[-1]] %*% beta[rows, j] 
    }
  }
  result_list[[j]] <- group_result
}

plot_single_series <- function(real_Y, est_Y, result, i) {
  # 提取非零结果
  non0_result <- result[sapply(result, function(x) !all(x == 0))]
  
  df <- data.frame(
    Time = rep(as.numeric(scaled_t), 2),
    Value = c(real_Y, est_Y),
    Type = rep(c("Real", "Estimated"), each = length(real_Y)),
    Series = rep(i, length(real_Y) * 2)
  )
  
  # 创建非零结果数据框，分别为每个子列表添加一条曲线
  nonzero_df <- do.call(rbind, lapply(seq_along(non0_result), function(idx) {
    if (idx == i) {
      data.frame(
        Time = as.numeric(scaled_t),
        Value = non0_result[[idx]],
        Type = "RedLine", # 特定子列表标记为红线
        Series = i
      )
    } else {
      data.frame(
        Time = as.numeric(scaled_t),
        Value = non0_result[[idx]],
        Type = paste0("NonZero_", idx), # 其他子列表标记为绿色线
        Series = i
      )
    }
  }))
  
  # 合并数据框
  combined_df <- rbind(df, nonzero_df)
  
  # 获取固定值
  intercept_value <- res$intercept[n_beta]
  
  # 使用 ggplot2 绘制图形
  p <- ggplot(combined_df, aes(x = Time, y = Value, color = Type, group = Type)) +
    geom_point(data = subset(combined_df, Type == "Real"), color = "blue", size = 1) + # 真实值为蓝点
    geom_line(data = subset(combined_df, Type == "Estimated"), color = "blue") +       # 估计值为蓝线
    geom_line(data = subset(combined_df, Type == "RedLine"), color = "red", linewidth = 0.8) + # 第 i 个子列表为红线
    geom_line(data = subset(combined_df, grepl("NonZero_", Type)), color = "green", linewidth = 0.8) 
  labs(title = paste("Series", i), x = "Time", y = "Value") +
    theme_minimal()
  p <- p + geom_segment(aes(x = min(scaled_t), xend = max(scaled_t), 
                            y = intercept_value, yend = intercept_value),
                        color = "yellow", linetype = "dashed", linewidth = 0.8)
  return(p)
}

plot_list <- list()

for (i in 1:ncol(y_est)) {
  plot_list[[i]] <- plot_single_series( real_Y=Y[, i], est_Y=y_est[, i],result=result_list[[i]], i=i)
}

combined_plot <- wrap_plots(plot_list, ncol = 2) + 
  plot_layout(ncol = 2, heights = rep(15, ceiling(ncol(Y)/2))) 

ggsave("D:/data/computer simulation/combined_plot_分解曲线.png", plot = combined_plot,width = 10,height = 32)




##########y拟合
plot_single_series <- function(real_Y, est_Y, i) {
  # 创建一个数据框，包含时间点、真实值和估计值
  df <- data.frame(Time = rep(as.numeric(scaled_t) , 2), 
                   Value = c(real_Y, est_Y), 
                   Type = rep(c("Real", "Estimated"), each = n), 
                   Series = rep(i, n * 2))
  
  # 使用ggplot2绘制图形
  p <- ggplot(df, aes(x = Time, y = Value, color = Type, group = Type)) +
    geom_point(data = subset(df, Type == "Real")) +
    geom_line(data = subset(df, Type == "Estimated")) +
    labs(title =  i, x = "Time", y = "Value") +
    theme_minimal()
  
  return(p)
}

plot_list <- list()

for (i in 1:ncol(y_est)) {
  plot_list[[i]] <- plot_single_series( real_Y=Y[, i], est_Y=y_est[, i], i=i)
}

combined_plot <- wrap_plots(plot_list, ncol = 2) + 
  plot_layout(ncol = 2, heights = rep(15, ceiling(ncol(Y)/2))) 

ggsave("D:/data/computer simulation/combined_plot.png", plot = combined_plot,width = 10,height = 32)

