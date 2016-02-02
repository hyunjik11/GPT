@everywhere using GPT_SGLD
@everywhere using DataFrames
@everywhere using PyPlot

#=
@everywhere Xtrain = readdlm("kin40k_train_data.txt", Float64);
@everywhere ytrain = readdlm("kin40k_train_labels.txt", Float64);
@everywhere Xtest = readdlm("kin40k_test_data.txt", Float64);
@everywhere ytest = readdlm("kin40k_test_labels.txt", Float64);


@everywhere D=8;
@everywhere seed=17;
@everywhere length_scale=1.5905;
@everywhere sigma_RBF=1.1812;
@everywhere signal_var=0.065;
@everywhere XtrainMean=mean(Xtrain,1); 
@everywhere XtrainStd=zeros(1,D);
@everywhere for i=1:D
	    XtrainStd[1,i]=std(Xtrain[:,i]);
	    end
@everywhere ytrainMean=mean(ytrain);
@everywhere ytrainStd=std(ytrain);
@everywhere Xtrain = datawhitening(Xtrain);
@everywhere ytrain=datawhitening(ytrain);
@everywhere Ntrain=length(ytrain);
@everywhere Ntest = size(Xtest,1);
@everywhere Xtest = (Xtest-repmat(XtrainMean,Ntest,1))./repmat(XtrainStd,Ntest,1);
@everywhere ytest = (ytest-ytrainMean)/ytrainStd;
@everywhere burnin=0;
@everywhere maxepoch=100;
@everywhere m=50;
=#
@everywhere n=8000;
@everywhere phitrain=featureNotensor(Xtrain,n,length_scale,sigma_RBF,seed);
@everywhere phitest=featureNotensor(Xtest,n,length_scale,sigma_RBF,seed);
@everywhere eps_theta=0.0002;
@everywhere decay_rate=0;
@everywhere sigma_theta=1;
#n=150: eps_theta=0.000005
#n=500: eps_theta=0.00003
#n=1000: eps_theta=0.00005
#n=2000: eps_theta=0.0001
#n=4000; eps_theta=0.0001
#n=8000; eps_theta=0.0001
#n=16000; eps_theta=0.0001

println("n=",n," m=",m," maxepoch=",maxepoch," eps_theta=",eps_theta," decay_rate=",decay_rate)
tic(); theta_store=GPNT_SGLD(phitrain,ytrain,signal_var,sigma_theta,m,eps_theta,decay_rate,burnin,maxepoch,1); toc();
numbatches=int(ceil(Ntrain/m))
    #trainRMSE=Array(Float64,maxepoch);
    testRMSE=Array(Float64,maxepoch);
    #fhat_train=Array(Float64,Ntrain,maxepoch);
    fhat_test=Array(Float64,Ntest,maxepoch);
    for epoch=1:maxepoch
        #fhat_train[:,i]=phitrain'*theta_store[:,2.5*n_samples+(i-1)*low];
        #fhat_test[:,i]=phitest'*theta_store[:,2.5*n_samples+(i-1)*low];
        #fhat_train[:,epoch]=phitrain'*theta_store[:,numbatches*epoch];
        #trainRMSE[epoch]=ytrainStd*sqrt(sum((fhat_train[:,epoch]-ytrain).^2)/Ntrain)
        fhat_test[:,epoch]=phitest'*theta_store[:,numbatches*epoch];
        testRMSE[epoch]=ytrainStd*sqrt(sum((fhat_test[:,epoch]-ytest).^2)/(Ntest));
    end
    #figure()
    #plot(trainRMSE)
    plot(testRMSE,label=string("n=",n))
	
	mean_fhat=mean(fhat_test[:,60:100],2);
	println("testRMSE with averaged pred=",ytrainStd*sqrt(sum((mean_fhat-ytest).^2)/Ntest));

