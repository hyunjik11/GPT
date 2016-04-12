f = read.csv("Folds5x2_pp.csv")

N=nrow(f)
D=ncol(f)-1
Ntrain=5000
for(i in 1:ncol(f)) {
  m = mean(f[1:Ntrain,i])
  s = sd(f[1:Ntrain,i])
  f[,i] = (f[,i] - m) / s
}

y=f[,D+1]
ytrain=y[1:Ntrain]
ytest=y[(Ntrain+1):N]

sigma=sqrt(0.0195)
sigma_RBF=sqrt(0.8333)
l=c(1.3978,0.0028,2.8966,7.5565)
#l=1.4332

n=100
Z=rnorm(n*D,0,1)/l #l can also be a vector (if using different l for each dim)
Z=t(matrix(Z,nrow=D))
b=2*pi*runif(n)
phi=matrix(0,N,n)
for(i in 1:N)  {
  phi[i,] = cos(as.matrix(f[i,1:4]) %*% t(Z)+t(b))
}
phi=sigma_RBF*sqrt(2/n)*phi
phitrain=phi[1:Ntrain,]
phitest=phi[(Ntrain+1):N,]
data=list(N=N,Ntrain=Ntrain,n=n,phitrain=phitrain,
phitest=phitest,ytrain=ytrain,ytest=ytest,sigma=sigma,ytrainStd=as.numeric(s))
library(ggplot2)
library(rstan)
#options(mc.cores = parallel::detectCores())
model = stan_model("fulltheta.stan")
fit = sampling(model, data=data, iter=100, chains=1)

out = extract(fit)
print(fit,"theta")
print(fit,"trainRMSE")
print(fit,"testRMSE")

