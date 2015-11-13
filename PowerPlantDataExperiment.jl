@everywhere using GPT_SGLD
@everywhere using Distributions
@everywhere using DataFrames
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
@everywhere burnin=20;
@everywhere maxepoch=5;
@everywhere Q=200;
@everywhere m=50;
@everywhere r=20;
@everywhere n=150;
@everywhere phitrain=feature(Xtrain,n,length_scale,seed);
@everywhere phitest=feature(Xtest,n,length_scale,seed);
@everywhere I=samplenz(r,D,Q,seed);
@everywhere epsw=100; 
@everywhere epsU=1e-15;
if 1==0
@everywhere t=Iterators.product(15:17,70:5:100)
@everywhere myt=Array(Any,21);
@everywhere it=1;
@everywhere for prod in t
	myt[it]=prod;
	it+=1;
end
    myRMSE=SharedArray(Float64,3*7);
    @parallel for  Tuple in myt
	i,j=Tuple;
        epsU=10.0^(-i); epsw=j;
	idx=int(3*(j-70)/5+i-14);
        tic();w_store,U_store=GPT_SGLDERM(phitrain,ytrain,sigma,I,r,Q,m,epsw,epsU,burnin,maxepoch);toc()
	tic();myRMSE=ytrainStd*RMSE(w_store,U_store,I,phitest,ytest);toc();
	
	println("RMSE=",myRMSE[idx],";seed=",seed,";sigma=",sigma, 		";length_scale=",length_scale,";n=",n,";r=",r,";Q=",Q,";m=",m,";epsw=",epsw, 		";epsU=",epsU,";burnin=",burnin,";maxepoch=",maxepoch);
    end
end
