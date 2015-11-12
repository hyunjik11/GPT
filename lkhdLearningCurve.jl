using GPT_SGLD
using Distributions
using DataFrames
using PyPlot

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
@everywhere Xtrain = GPT_SGLD.datawhitening(Xtrain);
@everywhere ytrain=GPT_SGLD.datawhitening(ytrain);
@everywhere Xtest = (data[Ntrain+1:end,1:D]-repmat(XtrainMean,N-Ntrain,1))./repmat(XtrainStd,N-Ntrain,1);
@everywhere ytest = (data[Ntrain+1:end,D+1]-ytrainMean)/ytrainStd;
@everywhere burnin=5;
@everywhere maxepoch=95;
@everywhere Q=200;
@everywhere m=50;
@everywhere r=20;
@everywhere n=150;
@everywhere phitrain=GPT_SGLD.feature(Xtrain,n,length_scale,seed);
@everywhere phitest=GPT_SGLD.feature(Xtest,n,length_scale,seed);
@everywhere I=samplenz(r,D,Q,seed);
epsU=1e-16; epsw=85;
	tic()
	w_store,U_store=GPT_SGLD.GPT_SGLDERM(phitrain,ytrain,sigma,I,r,Q,m,epsw,epsU,burnin,maxepoch);
	toc()
	T=size(w_store,2);
	low=int(floor(T/100));
	trainloglkhd=Array(Float64,100);
	testloglkhd=Array(Float64,100);
	for i=1:100
		fhat=pred(w_store[:,low*i],U_store[:,:,:,low*i],I,phitrain);
		trainloglkhd[i]=-(norm(ytrain-fhat))^2/(2*sigma^2);
	end
	figure()
	plot(trainloglkhd)
	title("Training loglikelihood")
	xlabel("percentage of total iterations")
	ylabel("loglikelihood")

	for i=1:100
		fhat=pred(w_store[:,low*i],U_store[:,:,:,low*i],I,phitest);
		testloglkhd[i]=-(norm(ytest-fhat))^2/(2*sigma^2);
	end
	figure()
	plot(testloglkhd)
	title("Test loglikelihood")
	xlabel("percentage of total iterations")
	ylabel("loglikelihood")
