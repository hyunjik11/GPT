@everywhere using GPT_SGLD
@everywhere using Distributions
@everywhere using DataFrames
#@everywhere using Iterators
@everywhere using PyPlot

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
@everywhere maxepoch=100;
@everywhere Q=200;
@everywhere m=50;
@everywhere r=20;
@everywhere n=150;
@everywhere phitrain=GPT_SGLD.feature(Xtrain,n,length_scale,seed);
@everywhere phitest=GPT_SGLD.feature(Xtest,n,length_scale,seed);
@everywhere I=samplenz(r,D,Q,seed);
@everywhere epsw=100; 
@everywhere epsU=1e-16;
w_store,U_store=GPT_SGLDERM(phitrain,ytrain,sigma,I,r,Q,m,epsw,epsU,burnin,maxepoch);
T=size(w_store,2);
trainloglkhd=SharedArray(Float64,T);
testloglkhd=SharedArray(Float64,T);
@sync @parallel for i=1:T
	fhat=pred(w_store[:,i],U_store[:,:,:,i],I,phitrain);
	trainloglkhd[i]=-(norm(ytrain-fhat))^2/(2*sigma^2);
end
figure()
plot(trainloglkhd)
titlename=string("LearningCurve. epsw=",epsw,",epsU=",epsU)
title(titlename)
xlabel("num_epochs-5(burnin)")
ylabel("loglikelihood")
@sync @parallel for i=1:T
	fhat=pred(w_store[:,i],U_store[:,:,:,i],I,phitest);
	testloglkhd[i]=-(norm(ytest-fhat))^2/(2*sigma^2);
end
plot(testloglkhd)
savefig("/homes/hkim/GPT/Plots/fineLearningCurve")

if 1==0
@everywhere t=Iterators.product(6:12,80:20:120)
@everywhere myt=Array(Any,21);
@everywhere it=1;
@everywhere for prod in t
	myt[it]=prod;
	it+=1;
end
@parallel for Tuple in myt
	i,j=Tuple;
	epsU=10.0^(-i); epsw=j;
	idx=int(7*(j-80)/20+i-5);
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
	titlename=string("LearningCurve. epsw=",epsw,",epsU=",epsU)
	title(titlename)
	xlabel("num_epochs")
	ylabel("loglikelihood")
	for i=1:100
		fhat=pred(w_store[:,low*i],U_store[:,:,:,low*i],I,phitest);
		testloglkhd[i]=-(norm(ytest-fhat))^2/(2*sigma^2);
	end
	plot(testloglkhd)
	filename=string("/homes/hkim/GPT/Plots/LearningCurveU",idx)
	savefig(filename)
end
end
