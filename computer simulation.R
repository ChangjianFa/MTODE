rm(list = ls())
library(splines,MASS)
library(grplasso) #install.packages('grplasso');
library(locfit)#install.packages('locfit');
library(fda)#install.packages('fda');
RandomSeed=11

#### Graph generation #####
# Generate a simple graph 
p<-4   # number of variables 

ks<-seq(from=1,to=p/2,length.out=p/2)
ks<-ks*pi*1.6
thetas<-matrix(0,p,p)
for(j in 1:{p/2}){
  thetas[ 2*j-1  ,2*j]<- ks[j]
  thetas[2*j, 2*j-1]<- {-ks[j]}
}
diag(thetas) <- c(0.6,0.8,1,1.2)
intercepts<-c(rep(0,p) )
thetas<-cbind(intercepts,thetas)
tgraph<-t(thetas[,-1]!=0)
image(tgraph)

#### Data generation #####
upp<-1 # range of time 
#R<-1; type_data="Replicates";
tgrid<-seq(from=0.01,to=upp,by=0.01)
# Find solutions of the ODE system given different initial values 
solutions<-list()
set.seed(RandomSeed) # Set seed for reproducibility 
# Data generation code
EulerM<-function(Initial_value=NULL, range=NULL, gap=0.01, f, theta,... ){
  T=(range[2]-range[1])/gap + 1
  P<-length(Initial_value)
  out_full<-matrix(0,T,P)
  out_full[1,]<-Initial_value
  
  for(i in 1:(T-1)){
    out_full[i+1,]<-out_full[i,]+f(out_full[i,],theta)*gap
  }
  return(out_full)
}

LinearODE<-function(X,theta){
  #theta=(alpha,beta,gamma, dealta)
  XPrime<- theta[,-1]%*%X+theta[,1]
  return(XPrime)
}
inis<-rnorm(p/2)
initialvalue<-c(rbind(sin(inis), cos(inis)))
Xs<-EulerM(initialvalue,range=c(0.01,upp),gap=0.01,f=LinearODE,theta=thetas)
solutions<-Xs

# plot
plot(0,0,xlim=c(0.01,upp),ylim=range(solutions))
for(j in 1:p){
  lines(solutions[,j]~tgrid,col=j)
}
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
library(mgcv)#拟合
library(nlme)#拟合
library(polynom)#多项式积分
DSIHT_Cpp <- function(x, y, weight, ic_type, ic_scale, sequence, kappa, g_index, ic_coef, method, coef1, coef2, eta, max_iter,nor) {
  .Call('_ADSIHT_DSIHT_Cpp', PACKAGE = 'ADSIHT', x, y, weight, ic_type, ic_scale, sequence, kappa, g_index, ic_coef, method, coef1, coef2, eta, max_iter,nor)
}
ADSIHT <- function(x, y, group,
                   s0, #####given L, s0 =  max(table(group))^(seq(1, L-1, length.out = L)/(L-1)), given L, donot need s0
                   kappa,######## 0.9-1
                   ic.type = c("dsic"),
                   ic.scale,###***** constant
                   ic.coef,####越小边的数量越多*****constant
                   L ,########
                   weight = rep(1, nrow(x)),
                   coef1 = 1,
                   coef2 = 1,
                   eta = 0.999,####越大越好0.5-1,method 选ols则不起作用
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

##生成观测值
getObservations<-function(X,times,sigma,gap){
  Y<-X[seq(1,nrow(X),length=n),]
  total_samples<-dim(Y)[1]*dim(Y)[2]
  noise<-matrix(rnorm(total_samples, mean=0,sd=sigma), nrow=dim(Y)[1], ncol =dim(Y)[2] )
  Y<-Y+noise
  return(Y)
}
n<- 20 #时间点数量
times<-tgrid[seq(1,nrow(solutions),length=n)] # 观测时间点
times_e<-seq(from=0.01,to=upp,by=0.01) # 估计时间点
observations<- cbind(times,getObservations(X=solutions,times=times,sigma=0.3,gap=0.01))

# smooth
type_smooth = "local polynomial" # use smooth splines to fit the smooth trajectories 
smoothX<-function(observations,times,times_e=times_e,deg,h=NULL,maxk=5000,
                  type_smooth=c("local polynomial","smoothspline" ,"linear interpolation")){
  Xhat<-list()
  gcvscore<-list()
  obs<-observations
  p<-dim(obs)[2]-1
  Xhat<-matrix(0,nrow=length(times),ncol= p)
  
  if( type_smooth=="linear interpolation") {
    for(i in 1:p){
      Xhat[,i]<-approx(x=obs[,1],y=obs[,i+1] ,xout=times_e,rule=2)$y
      gcvscore<-NULL
      Xprime<-NULL
      deg<-NULL
    }
  } else if(type_smooth=="local polynomial"){
    # Get the LP estimates 
    gcvscore<-matrix(0,nrow=length(h),ncol= p)
    for(i in 1:p){
      
      for(j in 1:length(h)){
        h_temp<-h[j]
        temp <- locfit(obs[,i+1]~lp(obs[,1],deg=deg,h=h_temp),maxk=maxk)
        gcvscore[j,i]<-gcv(obs[,i+1]~lp(obs[,1],deg=deg,h=h_temp),maxk=maxk)[4]
      }
      h_temp<-h[which.min(gcvscore[,i])]
      temp <- locfit(obs[,i+1]~lp(obs[,1],deg=deg,h= h_temp),maxk=maxk)
      Xhat[,i]<-predict(temp,newdata=times)
    }
    
  } else if (type_smooth == "smoothspline"){
    gcvscore<-numeric(p)
    for(i in 1:p){
      temp <- smooth.spline(obs[,1], obs[,i+1], all.knots = T)
      gcvscore[i]<-temp$cv.crit
      Xhat[,i]<-predict(temp,times)$y
    }
    
  }
  return(list(Xhat=Xhat,gcvscore=gcvscore,type_smooth=type_smooth))
  
}
smthed<-smoothX(observations=observations,times_e=times_e,times=times,deg=2,
                h=seq(min(observations[,1])/0.04, max(observations[,1])/2, length.out = 50),type_smooth=type_smooth)
xhat<-smthed$Xhat
colnames(xhat) <- paste0("p",1:p)
colnames(observations)[-1] <- paste0("p", 1:p)
colnames(solutions) <- paste0("p",1:p)

dat1<- pivot_longer(data.frame(x=times,y=xhat), cols = starts_with("y."), names_to = "group", values_to = "y")
dat2 <- pivot_longer(data.frame(x=times,y=observations[,-1]), cols = starts_with("y."), names_to = "group", values_to = "y")
dat3 <- pivot_longer(data.frame(x=tgrid,y=solutions), cols = starts_with("y."), names_to = "group", values_to = "y")
ggplot() +
  geom_line(data = dat1, aes(x, y), color = "red", linewidth = 1) +
  geom_point(data = dat2, aes(x, y), color = "black", size = 2) +
  geom_point(data = dat3, aes(x, y), color = "blue", size = 0.5) +
  facet_wrap(~ group, scales = "free_y") +
  theme_minimal() +
  labs(x = "x", y = "y") +
  theme(legend.position = "none")



A <- xhat
n <- nrow(A)
t <- ncol(A)
n_order= 1
group <- rep(1:(t*t), each=n_order+1)
ind_Leg0 <- c(0:(t-1))*(t*(n_order+1))+1 + rep(0:(t-1)) * (n_order+1)
Leg0 <- seq(1,length(group), by = n_order+1)
dep_Leg0 <- setdiff(Leg0, ind_Leg0)
group <- group[-dep_Leg0]

get_LOP_M <- function(x,n_order,times,name=NULL,scale=T){
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
  get_LOP_M(x=A[,edge],n_order,times=times)
}))
s_X <- scale(s_X,center=T,scale=F) #对自变量
X <- bdiag(rep(list(s_X), t))
X <- X[, -dep_Leg0]

Y <- observations[,-1]
s_Y <- scale(Y,center=T,scale=T)
Y <- matrix(as.vector(s_Y), nrow = n*t, ncol = 1)

res <- ADSIHT( x=X , y=Y, group = group, L = n_order+1, kappa = 0.999,ic.scale = 10,ic.coef = 1,)

# result
n_beta=which.min(res[["ic"]]) # n_beta=1
beta <- res$beta[,n_beta]
y_est <- X %*% beta + res$intercept[n_beta]

beta_star <- matrix(beta, nrow = (t*n_order)+1, ncol = t) #真实参数
beta_star
y_est <- matrix(y_est, nrow = n, ncol = t) #模拟变量
Y <- matrix(Y, nrow = n, ncol = t) 
cor(Y)

##########检查对y拟合
plot_single_series <- function(real_Y, est_Y, i) {
  df <- data.frame(Time = rep(as.numeric(times) , 2), 
                   Value = c(real_Y, est_Y))
  p <- ggplot(df, aes(x = Time, y = Value)) +
    geom_point(data = df[1:n,]) +
    geom_line(data = df[(n+1):(2*n),], color = "red")
  
  return(p)
}
plot_list <- list()
for (i in 1:ncol(y_est)) {
  plot_list[[i]] <- plot_single_series( real_Y=Y[, i], est_Y=y_est[, i], i=i)
}
combined_plot <- wrap_plots(plot_list, ncol = 2) + 
  plot_layout(ncol = 2, heights = rep(15, ceiling(ncol(Y)/2))) 
combined_plot
#beta_star[abs(beta_star) < 0.001] <- 0
beta_star







