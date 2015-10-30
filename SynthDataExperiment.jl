using GaussianProcess
using GPT_SGLD
using Distributions

sigmaRBF=1.4; sigma=0.2;
f=SECov(sigmaRBF,1)
gp=GP(0,f,4)
N=2000;
x1=rand(Uniform(0,50),N);
x2=rand(Uniform(30,100),N);
x3=rand(Uniform(1000,1020),N);
x4=rand(Uniform(50,100),N);
X=[x1 x2 x3 x4];
X=datawhitening(X);
y=GPrand(gp,X)+sigma*randn(N);
Xtrain=X[1:N/2,:]; ytrain=y[1:N/2];
Xtest=X[N/2+1:end,:]; ytest=y[N/2+1:end];
seed=17;
n=100;
sigmaRBF=1.4;
phitrain=feature(Xtrain,n,sigmaRBF,seed);
phitest=feature(Xtest,n,sigmaRBF,seed);
if 0==1
	r=10; Q=100;m=10;eps=0.001;maxepoch=1;
	tic()
	w_store,U_store,I=GPT_SGLDERM(phitrain,ytrain,sigma,sigmaRBF,r,Q,m,eps,eps,maxepoch);
	toc()
	minRMSE=RMSE(w_store,U_store,I,phitest,ytest)	
end


tic()
gp_post=GPpost(gp,Xtrain,ytrain,0.2);
gp_pred=Mean(gp_post,Xtest);
toc()
norm(ytest-gp_pred)/sqrt(N/2)

