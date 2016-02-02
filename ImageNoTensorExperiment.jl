#include("GPkit.jl-master/src/GPkit.jl")

@everywhere using GPT_SGLD
@everywhere using DataFrames: readdlm
@everywhere using PyPlot
#@everywhere using Iterators: product

@everywhere data=readdlm("segment.dat",Float64);
@everywhere data=data[:,[1,2,6:20]] #3rd column is constant,4th&5th columns are not useful - remove
@everywhere N=size(data,1);
@everywhere D=16;
@everywhere C=7;
@everywhere Ntrain=1300;
@everywhere Ntest=N-Ntrain;
@everywhere seed=17;
@everywhere length_scale=[5.1212,1.4029,5.0614,5.8121,6.1082,4.7774,1.7421,1.6444,1.8365,1.7417,1.8233,2.8132,1.2788,1.8477,2.0961,1.1489]; @everywhere sigma_RBF=11.4468
@everywhere Xtrain = data[1:Ntrain,1:D];
@everywhere ytrain = int(data[1:Ntrain,D+1]);
@everywhere XtrainMean=mean(Xtrain,1); 
@everywhere XtrainStd=zeros(1,D);
@everywhere for i=1:D
	    XtrainStd[1,i]=std(Xtrain[:,i]);
	    end
@everywhere Xtrain = datawhitening(Xtrain);
@everywhere Xtest = (data[Ntrain+1:Ntrain+Ntest,1:D]-repmat(XtrainMean,Ntest,1))./repmat(XtrainStd,Ntest,1);
@everywhere ytest = int(data[Ntrain+1:Ntrain+Ntest,D+1])
@everywhere burnin=20;
@everywhere maxepoch=100;
@everywhere m=50;
@everywhere n=150;
@everywhere phitrain=featureNotensor(Xtrain,n,length_scale,sigma_RBF,seed);
@everywhere phitest=featureNotensor(Xtest,n,length_scale,sigma_RBF,seed);
@everywhere eps_theta=0.001;
@everywhere decay_rate=0;
@everywhere sigma_theta=1;
#n=150: eps_theta=0.0003 eps_theta=0.001
#n=300: eps_theta=0.0005 eps_theta=0.0017
#n=500: eps_theta=0.0005 eps_theta=0.0017

println("n=",n," m=",m," maxepoch=",maxepoch," eps_theta=",eps_theta," decay_rate=",decay_rate)
tic(); theta_store=GPNT_SGLDclass(phitrain,ytrain,sigma_theta,m,eps_theta,decay_rate,burnin,maxepoch,2); toc();
numbatches=int(ceil(Ntrain/m))
    #trainRMSE=Array(Float64,maxepoch);
    prop_missed=Array(Float64,maxepoch);
	nlp=Array(Float64,Ntest);
	mean_nlp=Array(Float64,maxepoch);
    #fhat_train=Array(Float64,Ntrain,maxepoch);
    fhat_test=Array(Float64,Ntest,C,maxepoch);
	pred=Array(Integer,Ntest);
    for epoch=1:maxepoch
        #fhat_train[:,i]=phitrain'*theta_store[:,2.5*n_samples+(i-1)*low];
        #fhat_test[:,i]=phitest'*theta_store[:,2.5*n_samples+(i-1)*low];
        #fhat_train[:,epoch]=phitrain'*theta_store[:,numbatches*epoch];
        #trainRMSE[epoch]=ytrainStd*sqrt(sum((fhat_train[:,epoch]-ytrain).^2)/Ntrain)
        fhat_test[:,:,epoch]=phitest'*theta_store[:,:,numbatches*epoch];
		for i=1:Ntest
			pred[i]=indmax(fhat_test[i,:,epoch])
			nlp[i]=logsumexp(vec(fhat_test[i,:,epoch]))-fhat_test[i,ytest[i],epoch]
		end
        prop_missed[epoch]=1-sum(pred.==ytest)/Ntest
		mean_nlp[epoch]=mean(nlp)
    end
    #figure()
    #plot(trainRMSE)
    subplot(2,1,1); plot(prop_missed,label=string("n=",n))
	subplot(2,1,2); plot(mean_nlp,label=string("n=",n))
	mean_fhat=mean(fhat_test[:,:,60:100],3);
	for i=1:Ntest
		pred[i]=indmax(mean_fhat[i,:])
		nlp[i]=logsumexp(vec(mean_fhat[i,:]))-mean_fhat[i,ytest[i]]
	end
    prop_missed=1-sum(pred.==ytest)/Ntest
	mean_nlp=mean(nlp)
	println("prop_missed with averaged pred=",prop_missed);
	println("mean_nlp with averaged pred=",mean_nlp);


