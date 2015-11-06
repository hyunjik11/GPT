using GPT_SGLD
using Distributions
using DataFrames
using GaussianProcess

if 1==0
	data=readtable("Folds5x2_pp.csv", header = true);
	data = convert(Array,data);
	N=size(data,1); D=4; Ntrain=5000; n=100; seed=17; sigmaRBF=1.4332;sigma=0.2299;
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
	tic()
	phitrain=feature(Xtrain,n,sigmaRBF,seed);
	phitest=feature(Xtest,n,sigmaRBF,seed);
	toc()



function SDexp(phitrain,phitest,ytrain,ytest,seed,sigma,sigmaRBF,n,r,Q,m,epsw,epsU,burnin,maxepoch)
	D=size(Xtrain,2);
	sigmaw=sqrt(n^D/Q);
	tic()
	w_store,U_store,I=GPT_SGLDERM(phitrain,ytrain,sigma,sigmaw,r,Q,m,epsw,epsU,burnin,maxepoch);
	toc()
	minRMSE,minind=RMSE(w_store,U_store,I,phitest,ytest);
	println(" RMSE=",ytrainStd*minRMSE," begin_ind=",minind," epsw=", epsw, " epsU=",epsU);
	return w_store,U_store,I
end

end

if 1==1
tic()
f=SECov(sigmaRBF,1);
gp=GP(0,f,4);
gp_post=GPpost(gp,Xtrain,ytrain,sigma);
gp_pred=Mean(gp_post,Xtest);
toc()
println("RMSE for GP=",ytrainStd*norm(ytest-gp_pred)/sqrt(N-Ntrain))


r=10;Q=100;m=100;burnin=15;maxepoch=5;
for i=5:5:50
	for j=10:15
		epsw=i;epsU=10.0^(-j);
		SDexp(phitrain,phitest,ytrain,ytest,seed,sigma,sigmaRBF,n,r,Q,m,epsw,epsU,burnin,maxepoch);
	end
end

end




