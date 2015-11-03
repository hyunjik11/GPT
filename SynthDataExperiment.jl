using GaussianProcess
using GPT_SGLD
using Distributions
using HDF5


file="SynthData10000.h5";
Xtrain=h5read(file,"Xtrain");
XtrainMean=h5read(file,"XtrainMean");
XtrainStd=h5read(file,"XtrainStd");
ytrain=h5read(file,"ytrain");
Xtest=h5read(file,"Xtest");
ytest=h5read(file,"ytest");


if 1==0
tic()
gp_post=GPpost(gp,Xtrain,ytrain,0.2);
gp_pred=Mean(gp_post,Xtest);
toc()
println(norm(ytest-gp_pred)/sqrt(N-Ntrain))
end

function SDexp(seed,sigma,sigmaRBF,n,r,Q,m,epsw,epsU,maxepoch)
	D=size(Xtrain,1);
	sigmaw=sqrt(n^D/Q);
	phitrain=feature(Xtrain,n,sigmaRBF,seed);
	phitest=feature(Xtest,n,sigmaRBF,seed);
	#seed=17;sigma=0.2; sigmaRBF=1.4; n=10;r=5; Q=100;m=50;epsw=0.001;epsU=0.001;maxepoch=20;
	w_store,U_store,I=GPT_SGLDERM(phitrain,ytrain,sigma,sigmaw,r,Q,m,epsw,epsU,maxepoch);
	minRMSE,minind=RMSE(w_store,U_store,I,phitest,ytest);
	#fhat=pred(w_store[:,minind],U_store[:,:,:,minind],I,phitest);
	print("minRMSE=",minRMSE," minind=",minind);
	return w_store,U_store,I
end
