@everywhere using GPT_SGLD
@everywhere using DataFrames
@everywhere using PyPlot
@everywhere using Iterators
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
@everywhere burnin=60;
@everywhere maxepoch=10;
@everywhere Q=200;
@everywhere m=50;
@everywhere r=20;
@everywhere n=150;
@everywhere I=samplenz(r,D,Q,seed);
@everywhere scale=sqrt(n/(Q^(1/D)));
@everywhere phitrain=feature(Xtrain,n,length_scale,seed,scale);
@everywhere phitest=feature(Xtest,n,length_scale,seed,scale);
@everywhere epsw=1e-4; 
@everywhere epsU=1e-7;
tic();w_store,U_store=GPT_SGLDERMw(phitrain,ytrain,sigma,I,r,Q,m,epsw,burnin,maxepoch);toc()

if 1==0
@everywhere T=maxepoch*int(floor(Ntrain/m));
trainRMSE=SharedArray(Float64,T);
testRMSE=SharedArray(Float64,T);
trainfhat=SharedArray(Float64,Ntrain,1000);
testfhat=SharedArray(Float64,N-Ntrain,1000);
@sync @parallel for i=1:T
	fhattrain=pred(w_store[:,i],U_store[:,:,:,i],I,phitrain);
	trainRMSE[i]=ytrainStd*norm(ytrain-fhattrain)/sqrt(Ntrain);
	fhattest=pred(w_store[:,i],U_store[:,:,:,i],I,phitest);
	testRMSE[i]=ytrainStd*norm(ytest-fhattest)/sqrt(N-Ntrain);
	if i>T-1000
		trainfhat[:,i-(T-1000)]=fhattrain
		testfhat[:,i-(T-1000)]=fhattest
	end
end
println("epsw=",epsw," epsU=",epsU,"trainRMSE=",ytrainStd*norm(ytrain-mean(trainfhat,2))/sqrt(Ntrain))
println("epsw=",epsw," epsU=",epsU,"testRMSE=",ytrainStd*norm(ytest-mean(testfhat,2))/sqrt(N-Ntrain))
#plot(trainRMSE)
#plot(testRMSE)
#tic();myRMSEidx,temp=RMSE(w_store,U_store,I,phitest,ytest);toc();
end

if 1==0
@everywhere t=Iterators.product(6:8,4:5)
@everywhere myt=Array(Any,6);
@everywhere it=1;
@everywhere for prod in t
	myt[it]=prod;
        it+=1;
    end
#myRMSE=SharedArray(Float64,70);
@parallel for  Tuple in myt
	i,j=Tuple;
        epsU=10.0^(-i); epsw=10.0^(-j);
	#idx=int(3*(j-70)/5+i-14);
        w_store,U_store=GPT_SGLDERM(phitrain,ytrain,sigma,I,r,Q,m,epsw,epsU,burnin,maxepoch);
	myRMSE,temp=RMSE(w_store,U_store,I,phitest,ytest);
	myRMSE=ytrainStd*myRMSE;

	println("RMSE=",myRMSE,";seed=",seed,";sigma=",sigma,";length_scale=",length_scale,";n=",n,";r=",r,";Q=",Q,";m=",m,";epsw=",epsw,";epsU=",epsU,";burnin=",burnin,";maxepoch=",maxepoch);
    end
end

if 1==0
numI=50;
meanfhat=Array(Float64,N-Ntrain,numI);
for iseed=1:numI
	I=samplenz(r,D,Q,iseed);
	w_store,U_store=GPT_SGLDERM(phitrain,ytrain,sigma,I,r,Q,m,epsw,epsU,burnin,maxepoch);
	temp,meanfhat[:,iseed]=RMSE(w_store,U_store,I,phitest,ytest);
	println(iseed," iteration out of ",numI," done");
end
meanfhatfinal=mean(meanfhat,2);
println("RMSE=",norm(ytest-meanfhatfinal)/sqrt(N-Ntrain))
end

if 1==0 #storing variables to h5 file
using HDF5
c=h5open("SynthData1000.h5","w") do file
	write(file,"Xtrain",sdata(Xtrain));
	write(file,"XtrainMean",sdata(XtrainMean));
end
end

if 1==0 #reading variables from h5 file
using HDF5
file="10000SynthData.h5";
Xtrain=h5read(file,"Xtrain");
XtrainMean=h5read(file,"XtrainMean");
end

if 1==0 #writing stdout to file
    	outfile=open(filename,"a") #append to file
	println(outfile,"RMSE=",ytrainStd*predRMSE,";seed=",seed,";sigma=",sigma, 		";length_scale=",length_scale,";n=",n,";r=",r,";Q=",Q,";m=",m,";epsw=",epsw, 		";epsU=",epsU,";burnin=",burnin,";maxepoch=",maxepoch);
	close(outfile)
end



