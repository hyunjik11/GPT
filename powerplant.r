f = read.csv("../data/Folds5x2_pp.csv")

train = 1:50
test = length(train):nrow(f)
for(i in 1:ncol(f)) {
  m = mean(f[train,i])
  s = sd(f[train,i])
  f[,i] = (f[,i] - m) / s
}
#writeMat("../data/Folds5x2_pp.mat", data=f,sd=s)

randomfeatures = 50
w = rnorm(randomfeatures*4,0,1)
w = matrix(w,ncol=4)

X = matrix(0,nrow(f),randomfeatures*2)
for(i in 1:nrow(f))  {
  X[i,] = c(cos(as.matrix(f[i,1:4]) %*% t(w)),
            sin(as.matrix(f[i,1:4]) %*% t(w)))
}
data = list(X=X[train,],y=f$PE[train],n=length(train), 
            randomfeatures=randomfeatures, w=w,bw=.2,sigma=.1,d=4)  
library(rstan)
options(mc.cores = parallel::detectCores())
model = stan("powerplant.stan")

fit = stan(fit=model, data=data, iter=600, chains=2)

out = extract(fit)
print(fit,"beta")
print(fit,"bw")

# test

calc.yhat = function(bw,beta,w=data$w,x=data$x,n=data$n) {
  Xt = cbind(cos(X * bw), sin(X*bw))
  return(Xt %*% beta)
}
yhat = X %*% colMeans(out$beta) #calc.yhat(mean(data$bw),colMeans(out$beta))
plot(data$y,yhat)
sqrt(mean((f$PE[train] - yhat[train])^2))*s
sqrt(mean((f$PE[test] - yhat[test])^2))*s

