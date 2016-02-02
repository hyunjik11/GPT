@everywhere using GPT_SGLD
@everywhere using DataFrames
@everywhere using PyPlot

@everywhere data=DataFrames.readtable("Folds5x2_pp.csv", header = true);
@everywhere data = convert(Array,data);
@everywhere N=size(data,1);
@everywhere D=4;
@everywhere Ntrain=5000;
@everywhere Ntest=N-Ntrain;
@everywhere seed=17;
@everywhere length_scale=1.4332;
@everywhere signal_var=0.2299^2;
@everywhere sigma_RBF=1;
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
@everywhere ytrain= datawhitening(ytrain);
@everywhere Xtest = (data[Ntrain+1:Ntrain+Ntest,1:D]-repmat(XtrainMean,Ntest,1))./repmat(XtrainStd,Ntest,1);
@everywhere ytest = (data[Ntrain+1:Ntrain+Ntest,D+1]-ytrainMean)/ytrainStd;
@everywhere burnin=0;
@everywhere maxepoch=100;
@everywhere m=50;
@everywhere n=2000;
@everywhere phitrain=featureNotensor(Xtrain,n,length_scale,sigma_RBF,seed);
@everywhere phitest=featureNotensor(Xtest,n,length_scale,sigma_RBF,seed);
@everywhere eps_theta=0.00011;
@everywhere decay_rate=0;
@everywhere sigma_theta=1;
#n=150;eps_theta=0.0003;   0.00003
#n=500;eps_theta=0.00025;  0.00003
#n=1000;eps_theta=0.00025; 0.00003

    println("n=",n," m=",m," maxepoch=",maxepoch," eps_theta=",eps_theta," decay_rate=",decay_rate)
    tic(); theta_store=GPNT_SGLD(phitrain,ytrain,signal_var,sigma_theta,m,eps_theta,decay_rate,burnin,maxepoch,1); toc();
    #T=size(theta_store,2);
	numbatches=int(ceil(Ntrain/m))
    #n_samples=sample_epochs*integer(floor(Ntrain/m));
    #low=integer(floor(T/n_samples-5/2));
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
    #println("mintestRMSE=",minimum(testRMSE))
    #meanfhat_train=mean(fhat_train,2);
    #meanfhat_test=mean(fhat_test,2);
    #println(ytrainStd*sqrt(sum((meanfhat_train-ytrain).^2)/Ntrain))
    #println(ytrainStd*sqrt(sum((meanfhat_test-ytest).^2)/(N-Ntrain)))


if 1==0
    figure()
for i=1:10
    subplot(2,5,i)
    ax=gca()
    plot(vec(theta_store[10*i,:]))
    plot(vec(theta_store2[10*i,:]))
    ylim(-5,5)
    xlabel("num_iterations(no burnin)")
    setp(ax[:get_xticklabels](),fontsize=8)
end

savefig("Plots/NoTensorSGLDThetaTrace")
end

if 1==0
meangrad_store=SharedArray(Float64,n,50)
@parallel for i=1:50
theta_store,meangrad=GPNT_SGLD(phitrain,ytrain,sigma,sigma_theta,i*100,eps_theta,decay_rate,burnin,maxepoch,123);
meangrad_store[:,i]=meangrad;
println("m=",i*100," done")
end

end


