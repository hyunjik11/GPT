@everywhere using GPT_SGLD
@everywhere using DataFrames

@everywhere Xtrain = readdlm("kin40k_train_data.txt", Float64);
@everywhere ytrain = readdlm("kin40k_train_labels.txt", Float64);
@everywhere Xtest = readdlm("kin40k_test_data.txt", Float64);
@everywhere ytest = readdlm("kin40k_test_labels.txt", Float64);

@everywhere Ntrain=length(ytrain);
@everywhere D=8;
@everywhere seed=17;
@everywhere length_scale=3.1772;
@everywhere sigma=6.35346;
@everywhere sigma_RBF=0.686602;
@everywhere XtrainMean=mean(Xtrain,1); 
@everywhere XtrainStd=zeros(1,D);
@everywhere for i=1:D
	    XtrainStd[1,i]=std(Xtrain[:,i]);
	    end
@everywhere ytrainMean=mean(ytrain);
@everywhere ytrainStd=std(ytrain);
@everywhere Xtrain = datawhitening(Xtrain);
@everywhere ytrain=datawhitening(ytrain);
@everywhere Ntest = size(Xtest,1);
@everywhere Xtest = (Xtest-repmat(XtrainMean,Ntest,1))./repmat(XtrainStd,Ntest,1);
@everywhere ytest = (ytest-ytrainMean)/ytrainStd;
@everywhere burnin=0;
@everywhere maxepoch=50;
@everywhere Q=200;
@everywhere m=50;
@everywhere r=20;
@everywhere n=150;
@everywhere I=samplenz(r,D,Q,seed);
@everywhere scale=sigma_RBF*sqrt(n/(Q^(1/D)));
@everywhere phitrain=feature(Xtrain,n,length_scale,sigma_RBF,seed,scale);
@everywhere phitest=feature(Xtest,n,length_scale,sigma_RBF,seed,scale);
@everywhere epsw=1e-5;
@everywhere epsU=1e-8;

GPNT_hyperparameters(Xtrain,ytrain,n,1.0,1.0,0.2,seed)
