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

l1=1.4332 #length_scale of dim1
l2=1.4332 #length_scale of dim2
sigma=0.2299
sigma_RBF=1

n=100
Z1=rnorm(n,0,1)/l1
Z2=rnorm(n,0,1)/l2
b1=2*pi*runif(n)
b2=2*pi*runif(n)
phiU=matrix(0,N,n)
phiV=matrix(0,N,n)
for(i in 1:N)  {
  phiU[i,] = cos(t(Z1*f[i,1]+b1))
  phiV[i,] = cos(t(Z2*f[i,2]+b2))
}
phiU=sqrt(2/n)*phiU
phiV=sqrt(2/n)*phiV
phitrainU=phiU[1:Ntrain,]
phitrainV=phiV[1:Ntrain,]
phitestU=phiU[(Ntrain+1):N,]
phitestV=phiV[(Ntrain+1):N,]
r=5
data=list(N=N,Ntrain=Ntrain,n=n,r=r,phitrainU=phitrainU,phitrainV=phitrainV,
          phitestU=phitestU,phitestV=phitestV,ytrain=ytrain,ytest=ytest,
          sigma=sigma,ytrainStd=as.numeric(s))
library(ggplot2)
library(rstan)
options(mc.cores = parallel::detectCores())
model = stan_model("tensor2d.stan")
fit = sampling(model, data=data, iter=100, chains=4)

out = extract(fit)
print(fit,"trainRMSE")
print(fit,"testRMSE")
