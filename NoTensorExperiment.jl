@everywhere using GPT_SGLD
@everywhere using DataFrames
#@everywhere using PyPlot

@everywhere data=DataFrames.readtable("Folds5x2_pp.csv", header = true);
@everywhere data = convert(Array,data);
@everywhere N=size(data,1);
@everywhere D=4;
@everywhere Ntrain=5000;
@everywhere seed=17;
@everywhere length_scale=1.4332;
@everywhere sigma=0.2299;
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
@everywhere Xtest = (data[Ntrain+1:end,1:D]-repmat(XtrainMean,N-Ntrain,1))./repmat(XtrainStd,N-Ntrain,1);
@everywhere ytest = (data[Ntrain+1:end,D+1]-ytrainMean)/ytrainStd;
@everywhere Xtrain=Xtrain[1:Ntrain,:];
@everywhere burnin=0;
@everywhere maxepoch=500;
@everywhere m=50;
@everywhere n=150;
@everywhere phitrain=featureNotensor(Xtrain,n,length_scale,sigma_RBF,seed);
@everywhere phitest=featureNotensor(Xtest,n,length_scale,sigma_RBF,seed);
@everywhere eps_theta=0.00015;
@everywhere sample_epochs=20;
@everywhere decay_rate=0;
@everywhere sigma_theta=1;

if 1==0
    println("n=",n," m=",m," maxepoch=",maxepoch," sample_epochs=",sample_epochs," eps_theta=",eps_theta," decay_rate=",decay_rate)
    tic(); theta_store=GPNT_SGLD(phitrain,ytrain,sigma,sigma_theta,m,eps_theta,decay_rate,maxepoch,123); toc();
    T=size(theta_store,2);
    n_samples=sample_epochs*integer(floor(Ntrain/m));
    #low=integer(floor(T/n_samples-5/2));
    trainRMSE=Array(Float64,n_samples);
    testRMSE=Array(Float64,n_samples);
    fhat_train=Array(Float64,Ntrain,n_samples);
    fhat_test=Array(Float64,N-Ntrain,n_samples);
    for i=1:n_samples
        #fhat_train[:,i]=phitrain'*theta_store[:,2.5*n_samples+(i-1)*low];
        #fhat_test[:,i]=phitest'*theta_store[:,2.5*n_samples+(i-1)*low];
        fhat_train[:,i]=phitrain'*theta_store[:,end-n_samples+i];
        trainRMSE[i]=ytrainStd*sqrt(sum((fhat_train[:,i]-ytrain).^2)/Ntrain)
        fhat_test[:,i]=phitest'*theta_store[:,end-n_samples+i];
        testRMSE[i]=ytrainStd*sqrt(sum((fhat_test[:,i]-ytest).^2)/(N-Ntrain));
    end
    #figure()
    #plot(trainRMSE)
    #plot(testRMSE)
    println("mintestRMSE=",minimum(testRMSE))
    meanfhat_train=mean(fhat_train,2);
    meanfhat_test=mean(fhat_test,2);
    println(ytrainStd*sqrt(sum((meanfhat_train-ytrain).^2)/Ntrain))
    println(ytrainStd*sqrt(sum((meanfhat_test-ytest).^2)/(N-Ntrain)))
    
end

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


