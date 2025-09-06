rm(list = ls())
set.seed(123)
library(ggplot2)
library(reshape2)
library(SGL)
library(dplyr)
library(tidyr)
library(deSolve)
library(caret)
library(stringr)
library(foreach)
library(doParallel)
library(Matrix)
library(ADSIHT)
library(glmnet)
count_edge <- TRUE
t <- seq(0, 1, by = 0.01)
n <- length(t)
psi <- function(x) {
  c(sin(0.5*x),x^3)
}
n_order= 2
p=20
thetas<-matrix(0,n_order*p,p)
thetas[4, 1]<- 7
thetas[2, 2]<- -11
thetas[8, 3]<- 4
thetas[6, 4]<- -2
thetas[12, 5]<- -2
thetas[10, 6]<- 3

thetas[16, 7]<- 5
thetas[14, 8]<- -5
thetas[14, 9]<- 12
thetas[13, 10]<- 2
thetas[14, 11]<- -5
thetas[13, 12]<- 5
thetas[14, 13]<- -8
thetas[13, 14]<- -6

thetas[32, 15]<- -4
thetas[30, 16]<- 4
thetas[35, 17]<- -3
thetas[34, 18]<- 6
thetas[40, 19]<- -2
thetas[38, 20]<- 2

thetas[11, 2]<- 5
thetas[8, 5]<- -5
thetas[38, 17]<- -2
thetas[30, 20]<- -2

thetas[11, 9]<- -2
thetas[1, 11]<- 4
thetas[35, 13]<- 3
thetas[32, 14]<- -1

for(i in 1:p) {
  thetas[2*i-1, i] <- -2
}
mod1 <- function(Time, State, Pars) {
  with(as.list(c(State, Pars)), {
    State <- as.vector(sapply(State, psi))
    dx<- Pars%*%State
    return(list(dx))
  })
}

State <- round(rnorm(p, mean = 0, sd = 2), digits = 1)

times=t
out <- ode(y = State, times = times, 
           func = mod1, parms = t(thetas),
           method = "lsoda")[,-1]
