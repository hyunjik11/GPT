library(ggplot2)
library(rstan)
load('temp_final.RData')
library(h5); file=h5file("temp_eigenfeatures.h5"); phiU=t(file["phiU"][]); phiV=t(file["phiV"][]);
s=4.4960
r=10
data=list(N=N,Ntrain=Ntrain,n1=n1,n2=n2,r=r,phiU=phiU,phiV=phiV,
          indtrainU=indtrainU,indtrainV=indtrainV,indtestU=indtestU,indtestV=indtestV,
          ytrain=ytrain,ytest=ytest,sigma=sigma)

rstan_options(auto_write = TRUE)
options(mc.cores = 4)
model = stan_model("tensor2d_kron.stan")
#sink("tensor2d_10r.txt",append=TRUE)
numiter=200
warmup=100
ptm=proc.time()
fit = sampling(model, data=data, iter=numiter, chains=4,warmup=warmup)
#opt = optimizing(model,data=data)
#fit = vb(model,data=data)
time_elapsed=proc.time()-ptm
print(time_elapsed)



out = extract(fit)

trainRMSE=rep(0,numiter-warmup)
testRMSE=rep(0,numiter-warmup)
for (i in 1:(numiter-warmup)){
  id=(4*(i-1)+1):(4*i)
  yfittrain=colMeans(out$trainpred[id,])
  yfittest=colMeans(out$testpred[id,])
  trainRMSE[i]=sqrt(mean((ytrain-yfittrain)^2))*s
  testRMSE[i]=sqrt(mean((ytest-yfittest)^2))*s
}
cat("r=",r,"\n")
print("trainRMSE=")
print(trainRMSE)
print("testRMSE=")
print(testRMSE)
rhat=summary(fit)$summary[,"Rhat"];neff=summary(fit)$summary[,"n_eff"];
cat("rhat=",mean(rhat),"+/-",sd(rhat),";n_eff=",mean(neff),"+/-",sd(neff),"\n")
trainpred=colMeans(out$trainpred)
testpred=colMeans(out$testpred)
cat("final trainRMSE=",sqrt(mean((ytrain-trainpred)^2))*s,"\n")
cat("final testRMSE=",sqrt(mean((ytest-testpred)^2))*s,"\n")


#sink()


# library(R.matlab)
# df = readMat("/homes/hkim/TensorGP/src/uk_temp/temp.mat")
# attach(df)
# xtrain=as.matrix(xtrain)
# xtest=as.matrix(xtest)
# f = rbind(xtrain,xtest)
# ytrain=as.vector(ytrain)
# ytest=as.vector(ytest)
# 
# l=exp(hyp.cov[c(1,3)]);
# sigma_RBF = exp(hyp.cov[c(2,4)])
# sigma=exp(hyp.lik)[1]
# # find Ls,Lt
# L_func = function(x,l,sigma_RBF) {
#   if (length(dim(x)) >1) {
#     N=dim(x)[1]
#     K=matrix(0,N,N)
#     for (i in 1:N) {
#       for (j in 1:N) {
#         K[i,j] = sigma_RBF*exp(-1/l^2 * sum((x[i,] - x[j,])^2))
#       }
#     }}
#   else {N=length(x) 
#   K=matrix(0,N,N)
#   for (i in 1:N) {
#     for (j in 1:N) {
#       K[i,j] = sigma_RBF*exp(-1/l^2 * sum((x[i] - x[j])^2))
#     }
#   }}
#   return(t(chol(K,pivot=TRUE)))
# }
# space=unique(f[,1:2])
# n1=dim(space)[1]
# perm=sample(1:n1)
# space=space[perm,]
# temporal=unique(f[,3])
# temporal=as.matrix(temporal)
# n2=length(temporal)
# temporal=sample(temporal)
# 
# phiU=L_func(space,l[1],sigma_RBF[1])
# phiV=L_func(temporal,l[2],sigma_RBF[2])
# Ntrain=length(ytrain);Ntest=length(ytest)
# N=Ntrain+Ntest
# 
# indtrainU=vector(mode="integer",length=Ntrain)
# indtrainV=vector(mode="integer",length=Ntrain)
# indtestU=vector(mode="integer",length=Ntest)
# indtestV=vector(mode="integer",length=Ntest)
# 
# 
# for (i in 1:Ntrain) {
#   idu = which(apply(space,1,function(x) all(x== xtrain[i,1:2])));
#   indtrainU[i]=idu
# }
# indtrainV=match(xtrain[,3],temporal)
# for (i in 1:Ntest) {
#   idu = which(apply(space,1,function(x) all(x== xtest[i,1:2])));
#   indtestU[i]=idu
# }
# indtestV=match(xtest[,3],temporal)

