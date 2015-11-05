using GPT_SGLD
using HDF5

#if 1==0
file="10000SynthData.h5";
Xtrain=h5read(file,"Xtrain");
XtrainMean=h5read(file,"XtrainMean");
XtrainStd=h5read(file,"XtrainStd");
ytrain=h5read(file,"ytrain");
Xtest=h5read(file,"Xtest");
ytest=h5read(file,"ytest");
#end

if 1==0
tic()
gp_post=GPpost(gp,Xtrain,ytrain,0.2);
gp_pred=Mean(gp_post,Xtest);
toc()
println(norm(ytest-gp_pred)/sqrt(N-Ntrain))
end

function SDexp(seed,sigma,sigmaRBF,n,r,Q,m,epsw,epsU,burnin,maxepoch)
	D=size(Xtrain,2);
	sigmaw=sqrt(n^D/Q);
	phitrain=feature(Xtrain,n,sigmaRBF,seed);
	phitest=feature(Xtest,n,sigmaRBF,seed);
	w_store,U_store,I=GPT_SGLDERM(phitrain,ytrain,sigma,sigmaw,r,Q,m,epsw,epsU,burnin,maxepoch);
	minRMSE,minind=RMSE(w_store,U_store,I,phitest,ytest);
	#fhat=pred(w_store[:,minind],U_store[:,:,:,minind],I,phitest);
	println(" RMSE=",minRMSE," begin_ind=",minind," epsw=", epsw, " epsU=",epsU);
	return w_store,U_store,I
end

#seed=17;sigma=0.2;sigmaRBF=1.4;n=100;r=10;Q=100;m=100;epsw=10;epsU=1e-6; burnin=0;maxepoch=3;
#SDexp(seed,sigma,sigmaRBF,n,r,Q,m,epsw,epsU,burnin,maxepoch);

if 1==1
for i=-1:2
	for j=7:8
		seed=17;sigma=0.2;sigmaRBF=1.4;n=100;r=10;Q=100;m=100;epsw=10.0^(-i);epsU=10.0^(-j); burnin=5;maxepoch=5;
		SDexp(seed,sigma,sigmaRBF,n,r,Q,m,epsw,epsU,burnin,maxepoch);
	end
end
end
