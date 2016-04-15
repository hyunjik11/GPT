data = read.table("cadata.txt", col.names=  c( "MedHouseVal", "MedInc", "HouseAge", "AveRooms", "AveBedrms",
                                               "Population", "AveOccup", "long", "lat"))

data$MedHouseVal = log(data$MedHouseVal)
f=data.frame(data$long,data$lat,data$MedHouseVal)
N=nrow(f)
D=ncol(f)-1
Ntrain=10320

library(R.matlab)
library(ggplot2)
library(rstan)
perm = readMat("permutation.mat")
perm=c(perm[[1]])
f=f[perm,]

for(i in 1:ncol(f)) {
  m = mean(f[1:Ntrain,i])
  s = sd(f[1:Ntrain,i])
  f[,i] = (f[,i] - m) / s
}

y=f[,D+1]
ytrain=y[1:Ntrain]
ytest=y[(Ntrain+1):N]

sigma=0.3696#sqrt(0.0195)
sigma_RBF=0.9497#sqrt(0.8333)
#l=c(1.3978,0.0028,2.8966,7.5565)
l=c(0.0136,0.0216)

n=100

Z=rnorm(n*D,0,1)/l #l can also be a vector (if using different l for each dim)
Z=matrix(Z,nrow=D)
b=2*pi*runif(n)
phi=matrix(0,N,n)
for(i in 1:N)  {
  phi[i,] = cos(as.matrix(f[i,1:D]) %*% Z+t(b))
}
phi=sigma_RBF*sqrt(2/n)*phi
phitrain=phi[1:Ntrain,]
phitest=phi[(Ntrain+1):N,]
data=list(Ntrain=Ntrain,n=n,phitrain=phitrain,
ytrain=ytrain,sigma=sigma)

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

model = stan_model("fulltheta.stan")

sink("Routput_fulltheta.txt",append=TRUE)
cat("n=",n,"\n")
fit = sampling(model, data=data, iter=100, chains=4)
#opt = optimizing(model,data=data)
#fit = vb(model,data=data)

out = extract(fit)
print(fit,"theta")

fhat=phi%*% colMeans(out$theta)
#fhat=phi*opt$par["theta"]
trainRMSE=sqrt(mean((ytrain-fhat[1:Ntrain])^2))*s
testRMSE=sqrt(mean((ytest-fhat[(Ntrain+1):N])^2))*s
cat("trainRMSE=",trainRMSE,"\n")
cat("testRMSE=",testRMSE,"\n")
sink()
#plot(fit,plotfun="trace",pars="theta")

