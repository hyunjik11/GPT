library(mvtnorm)
require(MASS)
# devtools::install_github("cran/rstiefel")
#library(rstiefel)
#source('~/Dropbox/OxWaSP/GP/GPtensor/GPdraws.R')



##demo
N = 20;sigmaRBF=2
sequence = seq(0.5,by=1,length.out=N)
X = cbind(rep(sequence,N),rep(sequence,each=N))

#exact GP draws
par(mfrow=c(3,3))
par(mar=rep(1,4))
for (i in 1:9){
  y_mat = matrix(GPdraw(X,sigmaRBF),N,N)
  persp(sequence,sequence,y_mat,theta = 50, phi = 30, 
        expand = 0.5,xlab="x",ylab="y",zlab="GP exact",col=2,main="GP exact")
}
#approximate GP draws, fix r
r = 20
par(mfrow=c(3,3))
par(mar=rep(1,4))
for (n in c(200,160,130,100,80,60,40,30,20)){
  #   plot("",type="l",ylim=c(-5,5),xlim=c(1,100),main=paste("n=",n,", r=",r),ylab="f")
  y_mat = matrix(GPdrawsvd(X,n,r,sigmaRBF,2,sample(1:100)[1]),N,N)
  persp(sequence,sequence,y_mat,theta = 50, phi = 30, 
        expand = 0.5,xlab="x",ylab="y",zlab="GP approximate",col=2,main=paste("n=",n,", r=",r))
}
#approximate GP draws, fix n
n = 200
par(mfrow=c(3,3))
par(mar=rep(1,4))
for (r in c(200,150,100,80,50,20,10,5,1)){
  #   plot("",type="l",ylim=c(-5,5),xlim=c(1,100),main=paste("n=",n,", r=",r),ylab="f")
  y_mat = matrix(GPdrawsvd(X,n,r,sigmaRBF,2,sample(1:100)[1]),N,N)
  persp(sequence,sequence,y_mat,theta = 50, phi = 30, 
        expand = 0.5,xlab="x",ylab="y",zlab="GP approximate",col=2,main=paste("n=",n,", r=",r))
}
