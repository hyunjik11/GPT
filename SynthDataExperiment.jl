using Distributions
using GaussianProcess
using TGP
f=SECov(1.4,1)
gp=GP(0,f,4)
N=10000;
x1=rand(Uniform(0,50),N);
x2=rand(Uniform(30,100),N);
x3=rand(Uniform(1000,1020),N);
x4=rand(Uniform(50,100),N);
X=[x1 x2 x3 x4];
X=datawhitening(X);
y=GPrand(gp,X);
Xtrain=X[1:N/2,:]; ytrain=y[1:N/2];
Xtest=X[N/2+1:end,:]; ytest=y[N/2+1:end];
tic()
gp_post=GPpost(gp,Xtrain,ytrain,0.2);
pred=Mean(gp_post,Xtest);
toc()
norm(pred-ytest)/sqrt(N/2)

