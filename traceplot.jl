@everywhere using GPT_SGLD
@everywhere using Distributions
@everywhere using DataFrames
@everywhere using PyPlot
#@everywhere using Iterators
#using GaussianProcess

if 1==0
tic()
f=SECov(length_scale,1);
gp=GP(0,f,4);
gp_post=GPpost(gp,Xtrain,ytrain,sigma);
gp_pred=Mean(gp_post,Xtest);
toc()
println("RMSE for GP=",ytrainStd*norm(ytest-gp_pred)/sqrt(N-Ntrain))
end

@everywhere data=DataFrames.readtable("Folds5x2_pp.csv", header = true);
@everywhere data = convert(Array,data);
@everywhere N=size(data,1);
@everywhere D=4;
@everywhere Ntrain=5000;
@everywhere seed=17;
@everywhere length_scale=1.4332;
@everywhere sigma=0.2299;
@everywhere Xtrain = data[1:Ntrain,1:D];
@everywhere ytrain = data[1:Ntrain,D+1];
@everywhere XtrainMean=mean(Xtrain,1); 
@everywhere XtrainStd=zeros(1,D);
@everywhere for i=1:D
	    XtrainStd[1,i]=std(Xtrain[:,i]);
	    end
@everywhere ytrainMean=mean(ytrain);
@everywhere ytrainStd=std(ytrain);
@everywhere Xtrain = datawhitening(Xtrain);
@everywhere ytrain=datawhitening(ytrain);
@everywhere Xtest = (data[Ntrain+1:end,1:D]-repmat(XtrainMean,N-Ntrain,1))./repmat(XtrainStd,N-Ntrain,1);
@everywhere ytest = (data[Ntrain+1:end,D+1]-ytrainMean)/ytrainStd;
@everywhere burnin=5;
@everywhere maxepoch=200;
@everywhere Q=200;
@everywhere m=50;
@everywhere r=20;
@everywhere n=150;
@everywhere I=samplenz(r,D,Q,seed);
@everywhere scale=sqrt(n/(Q^(1/D)));
@everywhere phitrain=feature(Xtrain,n,length_scale,seed,scale);
@everywhere phitest=feature(Xtest,n,length_scale,seed,scale);
@everywhere epsw=5.5e-5; 
@everywhere epsU=1e-11;
tic();w_store,U_store=GPT_SGLDERM(phitrain,ytrain,sigma,I,r,Q,m,epsw,epsU,burnin,maxepoch);toc()
#pick 10 entries of w, and plot their traces
if 1==0
w=w_store[20:20:100,:];
U=U_store[1,1,:,:];
using HDF5
c=h5open("wU_store.h5","w") do file
	write(file,"w_store",w);
	write(file,"U_store",U);
end
figure();
for l=1:10
	plot(w_store[l,:]);
end
@everywhere T=maxepoch*int(floor(Ntrain/m));
trainRMSE=SharedArray(Float64,T);
testRMSE=SharedArray(Float64,T);
trainfhat=SharedArray(Float64,Ntrain,T);
testfhat=SharedArray(Float64,N-Ntrain,T);
@sync @parallel for i=1:T
	trainfhat[i]=pred(w_store[:,i],U_store[:,:,:,i],I,phitrain);
	trainRMSE[i]=ytrainStd*norm(ytrain-trainfhat[i])/sqrt(Ntrain);
	testfhat[i]=pred(w_store[:,i],U_store[:,:,:,i],I,phitest);
	testRMSE[i]=ytrainStd*norm(ytest-testfhat[i])/sqrt(N-Ntrain);
end
figure()
subplot(211); plot(trainRMSE)
subplot(212); plot(testRMSE)
end




