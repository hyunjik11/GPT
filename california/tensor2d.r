data = read.table("california/cadata.txt", col.names=  c( "MedHouseVal", "MedInc", "HouseAge", "AveRooms", "AveBedrms",
                                               "Population", "AveOccup", "lat", "long"))
data$MedHouseVal = log(data$MedHouseVal)
f=data.frame(data$lat,data$long,data$MedHouseVal)

N=nrow(f)
D=ncol(f)-1
Ntrain=10320

library(R.matlab)
library(ggplot2)
library(rstan)
perm = readMat("california/permutation.mat")
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
l1=0.0136 #lat
l2=0.0216 #long



n=5
r=2

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

data=list(N=N,Ntrain=Ntrain,n1=n,n2=n,r=r,phitrainU=phitrainU,phitrainV=phitrainV,
          phitestU=phitestU,phitestV=phitestV,ytrain=ytrain,
          sigma=sigma)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

model = stan_model("tensor2d.stan")
#sink("Routput_tensor.txt",append=TRUE)
cat("n=",n,"; r=",r,"\n")
fit = sampling(model, data=data, iter=10, chains=4)

out = extract(fit)
print(fit,"U")
print(fit,"V")
print(fit,"w")
trainpred=colMeans(out$trainpred)
testpred=colMeans(out$testpred)
trainRMSE=sqrt(mean((ytrain-trainpred)^2))*s
testRMSE=sqrt(mean((ytest-testpred)^2))*s
cat("trainRMSE=",trainRMSE,"\n")
cat("testRMSE=",testRMSE,"\n")
#sink()
