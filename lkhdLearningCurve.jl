@everywhere using GPT_SGLD
@everywhere using Distributions
@everywhere using DataFrames
@everywhere using Iterators
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
@everywhere maxepoch=10;
@everywhere Q=200;
@everywhere m=500;
@everywhere r=20;
@everywhere n=150;
@everywhere scale=sqrt(n/(Q^(1/D)));
@everywhere phitrain=GPT_SGLD.feature(Xtrain,n,length_scale,seed,scale);
@everywhere phitest=GPT_SGLD.feature(Xtest,n,length_scale,seed,scale);
@everywhere I=samplenz(r,D,Q,seed);

if 1==0
epsw=5.5e-5; 
epsU=1e-14;
w_store,U_store=GPT_SGD(phitrain,ytrain,sigma,I,r,Q,m,epsw,epsU,burnin,maxepoch);
@everywhere T=maxepoch*int(floor(Ntrain/m));
#trainloglkhd=SharedArray(Float64,T);
trainRMSE=SharedArray(Float64,T);
trainfhat=SharedArray(Float64,Ntrain,1000);
#testloglkhd=SharedArray(Float64,T);
testRMSE=SharedArray(Float64,T);
testfhat=SharedArray(Float64,N-Ntrain,1000);
@sync @parallel for i=1:T
	fhat=pred(w_store[:,i],U_store[:,:,:,i],I,phitrain);
	#trainloglkhd[i]=-(norm(ytrain-fhat))^2/(2*sigma^2);
	trainRMSE[i]=ytrainStd*norm(ytrain-fhat)/sqrt(Ntrain);
	if i>T-1000
		trainfhat[:,i-(T-1000)]=fhat
	end
end
#println("max trainloglkhd=",maximum(trainloglkhd))
println("epsw=",epsw," epsU=",epsU,"trainRMSE=",ytrainStd*norm(ytrain-mean(trainfhat,2))/sqrt(Ntrain))
figure()
plot(trainRMSE)
titlename=string("TrainingLearningCurve. epsw=",epsw,",epsU=",epsU)
title(titlename)
xlabel("num_iterations-burnin (100 per epoch)")
ylabel("RMSE")
@sync @parallel for i=1:T
	fhat=pred(w_store[:,i],U_store[:,:,:,i],I,phitest);
	#testloglkhd[i]=-(norm(ytest-fhat))^2/(2*sigma^2);
	testRMSE[i]=ytrainStd*norm(ytest-fhat)/sqrt(N-Ntrain);
	if i>T-1000
		testfhat[:,i-(T-1000)]=fhat
	end
end
println("epsw=",epsw," epsU=",epsU,"testRMSE=",ytrainStd*norm(ytest-mean(testfhat,2))/sqrt(N-Ntrain))
#println("max testloglkhd=",maximum(testloglkhd))
#println("epsw=",epsw," epsU=",epsU,"min testRMSE=",minimum(testRMSE))
plot(testRMSE)

savefig("/homes/hkim/GPT/Plots/RMSELearningCurve_SGD")

end

if 1==1
@everywhere t=Iterators.product(7:10,7:10)
@everywhere myt=Array(Any,16);
@everywhere it=1;
@everywhere for prod in t
	myt[it]=prod;
	it+=1;
end
@parallel for Tuple in myt
	i,j=Tuple;
	epsU=10.0^(-i);epsw=j*1e-5;
	idx=int(4*(j-7)+i-6);
	tic()
	w_store,U_store=GPT_SGLDERM(phitrain,ytrain,sigma,I,r,Q,m,epsw,epsU,burnin,maxepoch);
	toc()
	T=size(w_store,2);
	#trainRMSE=Array(Float64,T);
	testRMSE=Array(Float64,T);
	#for i=1:T
	#	fhat=pred(w_store[:,i],U_store[:,:,:,i],I,phitrain);
	#	trainRMSE[i]=ytrainStd*norm(ytrain-fhat)/sqrt(Ntrain);
	#end
	#figure()
	#subplot(211)
	#plot(trainRMSE)
	#titlename=string("Training RMSELearningCurve m=500. epsw=",epsw,",epsU=",epsU)
	#title(titlename)
	#xlabel("num_epochs-5burnin")
	#ylabel("RMSE")
	for i=1:T
		fhat=pred(w_store[:,i],U_store[:,:,:,i],I,phitest);
		testRMSE[i]=ytrainStd*norm(ytest-fhat)/sqrt(N-Ntrain);
	end
	println("mintestRMSE=",minimum(testRMSE)," epsU=",epsU," epsw=",epsw);
	#subplot(212)
	#plot(testRMSE)
	#title("Test RMSE")
	#filename=string("/homes/hkim/GPT/Plots/RMSELearningCurve500m",idx)
	#savefig(filename)
end
end
