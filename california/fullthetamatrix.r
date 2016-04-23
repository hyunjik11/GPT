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

sigma=0.3696
sigma_RBF=0.9497
l1=0.0136
l2=0.0216

n=5

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

data=list(N=N,Ntrain=Ntrain,n=n,phitrainU=phitrainU,phitrainV=phitrainV,
          phitestU=phitestU,phitestV=phitestV,ytrain=ytrain,
          sigma=sigma)
options(mc.cores = 4)
rstan_options(auto_write = TRUE)

model = stan_model("fullthetamatrix.stan")
sink("Routput_tensor.txt",append=TRUE)
cat("n=",n,"\n")
fit = sampling(model, data=data, iter=10, chains=1)

out = extract(fit)
print(fit,"theta")
trainpred=colMeans(out$trainpred)
testpred=colMeans(out$testpred)
trainRMSE=sqrt(mean((ytrain-trainpred)^2))*s
testRMSE=sqrt(mean((ytest-testpred)^2))*s
cat("trainRMSE=",trainRMSE,"\n")
cat("testRMSE=",testRMSE,"\n")
sink()
