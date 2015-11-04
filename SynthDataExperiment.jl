#using GPT_SGLD
#using HDF5

if 1==0
file="SynthData10000.h5";
Xtrain=h5read(file,"Xtrain");
XtrainMean=h5read(file,"XtrainMean");
XtrainStd=h5read(file,"XtrainStd");
ytrain=h5read(file,"ytrain");
Xtest=h5read(file,"Xtest");
ytest=h5read(file,"ytest");
end

if 1==0
tic()
gp_post=GPpost(gp,Xtrain,ytrain,0.2);
gp_pred=Mean(gp_post,Xtest);
toc()
println(norm(ytest-gp_pred)/sqrt(N-Ntrain))
end

function SDexp(seed,sigma,sigmaRBF,n,r,Q,m,a,epsw,epsU,maxepoch)
	D=size(Xtrain,2);
	sigmaw=sqrt((n/a)^D/Q);
	phitrain=feature(Xtrain,n,sigmaRBF,seed);
	phitest=feature(Xtest,n,sigmaRBF,seed);
	#seed=17;sigma=0.2; sigmaRBF=1.4; n=10;r=5; Q=100;m=50;epsw=0.001;epsU=0.001;maxepoch=20;
	w_store,U_store,I=GPT_SGLDERM(phitrain,ytrain,sigma,sigmaw,r,Q,m,a,epsw,epsU,maxepoch);
	minRMSE,minind=RMSE(w_store,U_store,I,phitest,ytest);
	#fhat=pred(w_store[:,minind],U_store[:,:,:,minind],I,phitest);
	println(" minRMSE=",minRMSE," minind=",minind," epsw=", epsw, " epsU=",epsU);
	return w_store,U_store,I
end

if 1==0
for i=0:6
	for j=4:8
		seed=17;sigma=0.2;sigmaRBF=1.4;n=100;r=10;Q=500;m=100;a=10;epsw=10.0^(-i);epsU=10.0^(-j); maxepoch=5;
		SDexp(seed,sigma,sigmaRBF,n,r,Q,m,a,epsw,epsU,maxepoch);
	end
end
end
