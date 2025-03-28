rm(list = ls())
library(Matrix)
library(deSolve)
library(orthopolynom)
library(ggplot2)
library(ADSIHT)
library(reshape2)
require(deSolve)
#### Graph Generation & Data Simulation ####
p <- 4   # number of variables (genes)

set.seed(1)
ks <- seq(from = 1, to = p/2, length.out = p/2)
ks <- ks * pi * 1.7
thetas <- matrix(0, p, p)
for(j in 1:(p/2)){
  thetas[2*j-1, 2*j] <- ks[j]
  thetas[2*j, 2*j-1] <- -ks[j]
}
diag(thetas) <- c(0.6, 0.8, 1, 1.2)
intercepts <- rep(0.2, p)
thetas <- cbind(intercepts, thetas)

upp <- 1   # time range
tgrid <- seq(from = 0, to = upp, by = 0.001)
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


#### Construct Legendre Basis for Each Gene ####
n_order <- 3  # order for Legendre polynomials
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

# For each gene (column of out), compute its Legendre basis and combine them
x_basis <- Reduce(cbind, lapply(1:p, function(xi) {
  get_LOP_M(x = out[, xi], n_order,times=tgrid)
}))

# Scale the basis matrix (predictors)
x_scaled <- scale(x_basis)
# Create a block-diagonal design matrix: each gene gets its own copy of the scaled basis
X <- as.matrix(bdiag(rep(list(x_scaled), p)))
X <- X[, -dep_Leg0]
#### Prepare Response ####
# We reshape the out into a column vector (stacking columns)
y_scaled <- scale(out)
Y <- matrix(as.vector(y_scaled), ncol = 1)

#### Fit the Model using ADSIHT ####
res <- ADSIHT(x = X, y = Y, group = group, L = n_order + 1, 
              kappa = 0.99, ic.scale = 1, ic.coef = 1)
n_beta <- which.min(res[["ic"]])  # choose the best model based on IC
beta <- res$beta[, n_beta]
order0_positions <- sapply(as.numeric(names(table(group)[table(group) == max(table(group))])), 
                           function(num) which(group == num)[1])
beta0_scaled <- res$intercept[n_beta]
# Reshape estimated beta into a matrix: each column corresponds to one gene's coefficients
beta_star <- data.frame(matrix(beta[-order0_positions], nrow = p * n_order, ncol = p))
beta_star <- rbind(beta[order0_positions],beta_star)
beta_star
# Plot the fitted scaled responses
y_est <- as.numeric(X %*% beta + beta0_scaled)
y_scaled_est_plot <- data.frame(times = rep(tgrid, p),y_est = y_est,obs=Y,
                                group = paste0("gene", rep(1:p, each = length(tgrid))))
ggplot(y_scaled_est_plot) +
  geom_line(mapping = aes(x = times, y = y_est, color = group)) +
  geom_point(mapping = aes(x = times, y = obs, color = group),size=0.1) +
  theme_minimal() + facet_wrap(~group)+
  labs(title = "Fitted Scaled Responses", x = "Time", y = "Scaled Response")

#### Recover Original Parameters ####
# Extract scaling attributes
sY <- attr(y_scaled, "scaled:scale")    # vector of length p for each gene (response scaling)
mY <- attr(y_scaled, "scaled:center")     # corresponding means of dX
sX <- attr(x_scaled, "scaled:scale")         # vector of length n_order for predictors
sX <- c(sX[1],sX[!grepl("leg0", names(sX))])
mX <- attr(x_scaled, "scaled:center")        # means of the predictors
mX <- c(mX[1],mX[!grepl("leg0", names(mX))])
# Recover original slopes for each gene
# For gene xi, each coefficient j is transformed as:
#    original_beta[j] = beta_star[j, xi] * (sd(dX[,xi]) / sd(x_scaled)[j])
BETA <- lapply(1:p, function(xi) {
  beta_orig <- beta_star[, xi] * (sY[xi] / sX)
  #beta_orig[beta_orig<1e-30]=0
  return(beta_orig)
})

# Recover original intercept for each gene:
# For gene xi:
#    original_intercept = beta0_scaled * sd(dX[,xi]) + mean(dX[,xi]) - sum( original_beta * mean(x_original) )
BETA0 <- lapply(1:p, function(xi) {
  orig_int <- beta0_scaled * sY[xi] + mY[xi] - sum(BETA[[xi]] * mX)
  return(orig_int)
})

# Display the true vs est
x_basis <- cbind(x_basis[,1],x_basis[,!grepl("leg0", colnames(x_basis))])
y_est = sapply(1:p, function(xi) x_basis %*% BETA[[xi]]+BETA0[[xi]])
plot_df = data.frame(x = rep(tgrid,p), y = as.vector(out), 
                     y_est = as.vector(y_est),
                     group = paste0("gene",rep(1:p,each = length(tgrid))))
ggplot(plot_df)+geom_point(mapping = aes(x=x,y=y))+
  geom_line(mapping = aes(x = x, y = y_est),color = 'red')+
  facet_wrap(~group)+theme_bw()


#we further assume intercept is in independent effect
B = t(cbind(unlist(BETA0),t(Reduce(cbind,BETA))))
y_est = cbind(1,x_basis)%*%B




j=3;n=30;sd=0.5
#show results
test_x <- function(j,n=30,sd = 0.1){
  idx = seq(3,(p*n_order+2))
  idx = split(idx, cut(seq_along(idx), breaks = p, labels = FALSE))
  idx[[j]] = c(1,2,idx[[j]])
  
  x_basis2 = cbind(1,x_basis)
  #ture effect
  out2 = cbind(tgrid,rowSums(sapply(1:p,function(xi) x_basis2[,idx[[xi]]]%*% B[idx[[xi]],j])),
               sapply(1:p,function(xi) x_basis2[,idx[[xi]]]%*% B[idx[[xi]],j]))
  
  trans_out <- function(out,j){
    id = paste0("gene",j)
    
    out = as.data.frame(out)
    colnames(out) = c('time',"obs",paste0("gene",1:p))
    out = melt(out,id.vars = 'time')
    out$group = 'Dep'
    out$group[out$variable == id] = 'Ind'
    out$group[out$variable == 'obs'] = 'obs'
    out
  }
  out2 = trans_out(out2,j)
  
  p1 = ggplot(out2) + 
    geom_line(mapping = aes(x=time,y = value,group = variable,color = group))+
    scale_color_manual(values = c(obs = 'blue',
                                  Ind = 'red',
                                  Dep = 'green'))+
    theme_minimal()
  p1
  
  
  #### Data generation ####
  #all_pars = lapply(1:p, function(xi) 
  #  split(BETA[[xi]], cut(seq_along(BETA[[xi]]), breaks = p, labels = FALSE))
  #)
  data = out[seq(1,nrow(out), length.out = n),] + matrix(rnorm(n*p, sd = sd),n,p,byrow = T)
  times = tgrid[seq(1,nrow(out), length.out = n)]
  
  ODEsolve <- function(data,times=times,n_order=n_order){
    if(is.null(times)){
      times = 1:nrow(data)
    }
    p = ncol(data)
    #smooth data
    fit <- lapply(1:p, function (xi) 
      smooth.spline(times, as.numeric(data[,xi]), spar = 0.5))
    
    x_smooth <- sapply(1:p, function(xi)  predict(fit[[xi]], x = times)$y)
    
    plot_df = data.frame(x = rep(times,p), obs = as.vector(data), 
                         x_smooth = as.vector(x_smooth),
                         x_ture = as.vector(out[seq(1,nrow(out), length.out = n),]),
                         group = paste0("gene",rep(1:p,each = length(times))))
    ggplot(plot_df)+geom_point(mapping = aes(x=x,y=obs))+
      geom_line(mapping = aes(x = x, y = x_smooth),color = 'red')+
      geom_line(mapping = aes(x = x, y = x_ture),color = 'blue')+
      facet_wrap(~group)+theme_bw()
    
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
    
    res <- ADSIHT(x = X, y = Y, group = group, L = n_order, 
                  kappa = 0.99, ic.scale = 1, ic.coef = 1)
    n_beta <- which.min(res[["ic"]])
    beta <- res$beta[, n_beta]
    order0_positions <- sapply(as.numeric(names(table(group)[table(group) == max(table(group))])), 
                               function(num) which(group == num)[1])
    beta0_scaled <- res$intercept[n_beta]
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
    x_basis <- cbind(x_basis[,1],x_basis[,!grepl("leg0", colnames(x_basis))])
    x_basis_est = cbind(1,x_basis)
    
    
    return(list(x_basis = x_basis_est,
                BETA_est = B_est,
                x_smooth = x_smooth)) 
  }
  
  
  res = ODEsolve(data = data,times=times, n_order = n_order)
  
  
  
  B_est = res$BETA_est
  #B_est[B_est<1e-10] = 0
  x_basis_est = res$x_basis
  
  
  out3 <- cbind(times,rowSums(sapply(1:p,function(xi) x_basis_est[,idx[[xi]]]%*% B_est[idx[[xi]],j])),
                sapply(1:p,function(xi) x_basis_est[,idx[[xi]]]%*% B_est[idx[[xi]],j]))
  out3 <- trans_out(out3,j)
  
  
  
  p2 = ggplot() + geom_point(mapping = aes(x=times, y = data[,j]),color = 'blue')+
    geom_line(out2, mapping = aes(x=time,y = value,group = variable,color = group))+
    geom_line(out3, mapping = aes(x=time,y = value,group = variable,color = group),linetype = 2)+
    scale_color_manual(values = c(obs = 'blue',
                                  Ind = 'red',
                                  Dep = 'green'))+
    xlab("Time") + ylab("Gene Expression Level") + ggtitle(paste0("gene",j))+
    theme_minimal()
  
  p2
  
  
  
  # Convert true parameters to a data frame and melt it into long format with a "True" label
  B <- data.frame(B)
  colnames(B) <- paste0("gene", 1:p)
  B_long <- melt(B, variable.name = "gene", value.name = "value")
  B_long$group <- "True"
  
  # Convert estimated parameters to a data frame and melt it into long format with an "Estimated" label
  B_est <- data.frame(B_est)
  colnames(B_est) <- paste0("gene", 1:p)
  B_est_long <- melt(B_est, variable.name = "gene", value.name = "value")
  B_est_long$group <- "Estimated"
  
  # Combine the true and estimated data frames
  B_plot <- rbind(B_long, B_est_long)
  
  # Plot using ggplot with facet_wrap and different colors for True and Estimated groups
  p3 = ggplot(B_plot, aes(x = group, y = value, color = group)) +
    geom_point(size = 3, position = position_dodge(width = 0.3)) +
    facet_wrap(~ gene, scales = "free_y") +
    labs(title = "Comparison of True and Estimated Parameters by Gene",
         x = "Parameter Group",
         y = "Parameter Value") +
    theme_minimal()
  
  
  pp = wrap_plots(p2,p3) 
  pp
}


test_x(4,n=30 ,sd = 0.1)










