#include("GPkit.jl-master/src/GPkit.jl")

@everywhere using GPT_SGLD
@everywhere using DataFrames: readdlm
#@everywhere using GPkit
#@everywhere using PyPlot
#@everywhere using Iterators: product

@everywhere data=readdlm("segment.dat",Float64);
@everywhere data=data[:,[1,2,4:20]] #3rd column is constant - remove
@everywhere N=size(data,1);
@everywhere D=18;
@everywhere Ntrain=1300;
@everywhere Ntest=N-Ntrain;
@everywhere seed=17;
@everywhere length_scale=1.4332;
@everywhere sigma_RBF=1;
@everywhere signal_var=0.2299^2;
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
@everywhere Xtest = (data[Ntrain+1:Ntrain+Ntest,1:D]-repmat(XtrainMean,Ntest,1))./repmat(XtrainStd,Ntest,1);
@everywhere ytest = (data[Ntrain+1:Ntrain+Ntest,D+1]-ytrainMean)/ytrainStd;
@everywhere burnin=0;
@everywhere maxepoch=200;
@everywhere Q=200;
@everywhere m=50;
@everywhere r=10;
@everywhere n=150;
@everywhere I=samplenz(r,D,Q,seed);
@everywhere scale=sqrt(n/(Q^(1/D)));
@everywhere phitrain=feature(Xtrain,n,length_scale,sigma_RBF,seed,scale);
@everywhere phitest=feature(Xtest,n,length_scale,sigma_RBF,seed,scale);
@everywhere epsw=1e-4; 
@everywhere epsU=1e-7;
@everywhere epsilon=1e-8;
@everywhere alpha=0.99;
@everywhere L=30;
@everywhere param_seed=234;
