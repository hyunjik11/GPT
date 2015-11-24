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
@everywhere maxepoch=5000;
@everywhere Q=100;
@everywhere m=5000;
@everywhere r=20;
@everywhere n=150;
@everywhere scale=sqrt(n/(Q^(1/D)));
@everywhere phitrain=GPT_SGLD.feature(Xtrain,n,length_scale,seed,scale);
@everywhere phitest=GPT_SGLD.feature(Xtest,n,length_scale,seed,scale);
@everywhere I=samplenz(r,D,Q,seed);

if 1==1
epsw=5.5e-5; 
epsU=1e-12;
tic();
w_store,U_store=GPT_SGLDERM(phitrain,ytrain,sigma,I,r,Q,m,epsw,epsU,burnin,maxepoch);
toc();
@everywhere T=maxepoch*int(floor(Ntrain/m));
low=int(floor(T/100));
trainRMSE=Array(Float64,100);
testRMSE=Array(Float64,100);
#trainloglkhd=SharedArray(Float64,T);
#testloglkhd=SharedArray(Float64,T);
for i=1:100
	fhat=pred(w_store[:,low*i],U_store[:,:,:,low*i],I,phitrain);
	#trainloglkhd[i]=-(norm(ytrain-fhat))^2/(2*sigma^2);
	trainRMSE[i]=ytrainStd*norm(ytrain-fhat)/sqrt(Ntrain);
end
#println("max trainloglkhd=",maximum(trainloglkhd))
#println("epsw=",epsw," epsU=",epsU,"trainRMSE=",ytrainStd*norm(ytrain-mean(trainfhat,2))/sqrt(Ntrain))
#figure()
#plot(trainRMSE)
#titlename=string("LearningCurveNominibatch. epsw=",epsw,",epsU=",epsU)
#title(titlename)
#xlabel("num_epochs/50")
#ylabel("RMSE")
for i=1:100
	fhat=pred(w_store[:,low*i],U_store[:,:,:,low*i],I,phitest);
	testRMSE[i]=ytrainStd*norm(ytest-fhat)/sqrt(N-Ntrain);
end
#println("epsw=",epsw," epsU=",epsU,"testRMSE=",ytrainStd*norm(ytest-mean(testfhat,2))/sqrt(N-Ntrain))
#println("max testloglkhd=",maximum(testloglkhd))
#println("epsw=",epsw," epsU=",epsU,"min testRMSE=",minimum(testRMSE))
#plot(testRMSE)

#savefig("/homes/hkim/GPT/Plots/RMSELearningCurve_nominibatchLong")

end

if 1==0
#@everywhere t=Iterators.product(9:14,5:10:45)
#@everywhere myt=Array(Any,30);
#@everywhere it=1;
#@everywhere for prod in t
#	myt[it]=prod;
#	it+=1;
#end
@parallel for j=1:10
	#i,j=Tuple;
	epsU=j*1e-11;epsw=5.5e-5;
	#idx=int(2*(j-10)+i-9);
	tic();w_store,U_store=GPT_SGLDERM(phitrain,ytrain,sigma,I,r,Q,m,epsw,epsU,burnin,maxepoch); toc();
	#myRMSE,temp=RMSE(w_store,U_store,I,phitest,ytest);
	T=maxepoch*int(floor(Ntrain/m));
	low=int(floor(T/100));
	trainRMSE=Array(Float64,100);
	testRMSE=Array(Float64,100);
	for i=1:100
		fhat=pred(w_store[:,low*i],U_store[:,:,:,low*i],I,phitrain);
		trainRMSE[i]=ytrainStd*norm(ytrain-fhat)/sqrt(Ntrain);
	end
	figure()
	subplot(211)
	plot(trainRMSE)
	titlename=string("Training RMSELearningCurve. epsw=",epsw,",epsU=",epsU)
	title(titlename)
	xlabel("num_epochs-5burnin")
	ylabel("RMSE")
	for i=1:100
		fhat=pred(w_store[:,low*i],U_store[:,:,:,low*i],I,phitest);
		testRMSE[i]=ytrainStd*norm(ytest-fhat)/sqrt(N-Ntrain);
	end
	#println("testRMSE=",ytrainStd*myRMSE," epsU=",epsU," epsw=",epsw);
	subplot(212)
	plot(testRMSE)
	title("Test RMSE")
	filename=string("/homes/hkim/GPT/Plots/RMSELearningCurveUfine",j)
	savefig(filename)
end
end
