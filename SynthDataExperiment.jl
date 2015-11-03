using GaussianProcess
using GPT_SGLD
using Distributions

if 1==0
sigmaRBF=1.4; sigma=0.2;
f=SECov(sigmaRBF,1)
gp=GP(0,f,4)
N=1000; D=4; Ntrain=500;
x1=rand(Uniform(0,50),N);
x2=rand(Uniform(30,100),N);
x3=rand(Uniform(1000,1020),N);
x4=rand(Uniform(50,100),N);
X=[x1 x2 x3 x4];
Xtrain = X[1:Ntrain,1:D];
XtrainMean=mean(Xtrain,1); 
XtrainStd=Array(Float64,1,D);
for i=1:D
    XtrainStd[1,i]=std(Xtrain[:,i]);
end
Xtrain = datawhitening(Xtrain);
Xtest = (X[Ntrain+1:end,1:D]-repmat(XtrainMean,N-Ntrain,1))./repmat(XtrainStd,N-Ntrain,1);
X=[Xtrain;Xtest];
y=GPrand(gp,X)+sigma*randn(N);
ytrain = y[1:Ntrain];
ytest = y[Ntrain+1:end];

tic()
gp_post=GPpost(gp,Xtrain,ytrain,0.2);
gp_pred=Mean(gp_post,Xtest);
toc()
println(norm(ytest-gp_pred)/sqrt(N-Ntrain))
end

#if 1==0
seed=17;
n=100;
phitrain=feature(Xtrain,n,sigmaRBF,seed);
phitest=feature(Xtest,n,sigmaRBF,seed);
r=5; Q=100;m=50;eps=0.01;maxepoch=10;
tic()
w_store,U_store,I=GPT_SGLDERM(phitrain,ytrain,sigma,sigmaRBF,r,Q,m,eps,eps,maxepoch);
toc()
minRMSE,minind=RMSE(w_store,U_store,I,phitest,ytest);
tic()
fhat=pred(w_store[:,minind],U_store[:,:,:,minind],I,phitest);
toc()	
println(minRMSE)
#end
