#using GPT_SGLD
#using Distributions
#using DataFrames

#if 1==0
	data=readtable("Folds5x2_pp.csv", header = true);
	data = convert(Array,data);
	N=size(data,1); D=4; Ntrain=5000;
	Xtrain = data[1:Ntrain,1:D]; ytrain = data[1:Ntrain,D+1];
	XtrainMean=mean(Xtrain,1); 
	XtrainStd=Array(Float64,1,D);
	for i=1:D
	    XtrainStd[1,i]=std(Xtrain[:,i]);
	end
	ytrainMean=mean(ytrain); ytrainStd=std(ytrain);
	Xtrain = datawhitening(Xtrain); ytrain=datawhitening(ytrain);
	Xtest = (data[Ntrain+1:end,1:D]-repmat(XtrainMean,N-Ntrain,1))./repmat(XtrainStd,N-Ntrain,1);
	ytest = (data[Ntrain+1:end,D+1]-ytrainMean)/ytrainStd;
	seed=17;
	n=100;
	sigmaRBF=1.4332; sigma=0.2299;
	phitrain=feature(Xtrain,n,sigmaRBF,seed);
	phitest=feature(Xtest,n,sigmaRBF,seed);
#end

r=20; Q=100;m=100;eps=0.00001;maxepoch=10;
tic()
w_store,U_store,I=GPT_SGLDERM(phitrain,ytrain,sigma,sigmaRBF,r,Q,m,eps,eps,maxepoch);
toc()
minRMSE,minind=RMSE(w_store,U_store,I,phitest,ytest);
tic()
fhat=pred(w_store[:,minind],U_store[:,:,:,minind],I,phitest);
toc()
testRMSE=minRMSE*ytrainStd



