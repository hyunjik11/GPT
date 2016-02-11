@everywhere using GPT_SGLD
@everywhere using PyPlot
@everywhere using HDF5
#@everywhere using Iterators

@everywhere file="TensorSynthData8000N.h5";
@everywhere X=h5read(file,"X");
@everywhere y=h5read(file,"y3"); @everywhere signal_var=1e-3;
@everywhere w=h5read(file,"w");
@everywhere U=h5read(file,"U");
@everywhere I=h5read(file,"I");
@everywhere phi=h5read(file,"phi");
@everywhere N=size(X,1);
@everywhere D=5;
@everywhere Ntrain=int(N/2);
@everywhere Ntest=N-Ntrain;
@everywhere seed=18;
@everywhere length_scale=[1-2/D,1-1/D,1,1+1/D,1+2/D];
@everywhere sigma_RBF=1;
@everywhere Xtrain = X[1:Ntrain,:];
@everywhere ytrain = y[1:Ntrain];
@everywhere XtrainMean=mean(Xtrain,1); 
@everywhere XtrainStd=zeros(1,D);
@everywhere for i=1:D
	    XtrainStd[1,i]=std(Xtrain[:,i]);
	    end
@everywhere ytrainMean=mean(ytrain);
@everywhere ytrainStd=std(ytrain);
@everywhere Xtrain = datawhitening(Xtrain);
@everywhere ytrain=datawhitening(ytrain);
@everywhere Xtest = (X[Ntrain+1:N,:]-repmat(XtrainMean,Ntest,1))./repmat(XtrainStd,Ntest,1);
@everywhere ytest = (y[Ntrain+1:N]-ytrainMean)/ytrainStd;
@everywhere burnin=0;
@everywhere maxepoch=100;
@everywhere Q=200;
@everywhere m=50;
@everywhere r=20;
@everywhere n=150;
@everywhere scale=sqrt(n/(Q^(1/D)));
#@everywhere phitrain=featureNotensor(Xtrain,n,length_scale,sigma_RBF,seed); @everywhere phitest=featureNotensor(Xtest,n,length_scale,sigma_RBF,seed);
#@everywhere phitrain=feature(Xtrain,n,length_scale,sigma_RBF,seed,scale); @everywhere phitest=feature(Xtest,n,length_scale,sigma_RBF,seed,scale);
@everywhere phitrain=phi[:,:,1:Ntrain]; @everywhere phitest=phi[:,:,Ntrain+1:end];
@everywhere epsw=1e-5; 
@everywhere epsU=1e-8;
@everywhere epsilon=1e-5;
@everywhere decay_rate=0;
@everywhere alpha=0.99;
@everywhere L=30;
@everywhere param_seed=1;
@everywhere sigma_theta=1;

#=
tic(); theta_store=GPNT_SGLD(phitrain, ytrain, signal_var, sigma_theta, m, epsilon,decay_rate, burnin, maxepoch, param_seed); toc();
testRMSE=Array(Float64,maxepoch)
finalpred=zeros(Ntest)
numbatches=int(ceil(Ntrain/m))
for epoch=1:maxepoch
	testpred=phitest'*theta_store[:,epoch*numbatches];
    #testpred=pred(w_store[:,epoch*numbatches],U_store[:,:,:,epoch*numbatches],I,phitest)
	if (maxepoch-epoch)<50
		finalpred+=testpred
	end
    testRMSE[epoch]=ytrainStd*norm(ytest-testpred)/sqrt(Ntest)
end
#plot(testRMSE,label=string("n=",n))
finalpred/=50
println("n=",n,";epsilon=",epsilon,";vanilla SGLD RMSE over last 50 epochs=",ytrainStd*norm(ytest-finalpred)/sqrt(Ntest))
=#
#=
#for param_seed=4:5
tic(); w_store,U_store=GPTregression(phitrain, ytrain, signal_var, I, r, Q, m, epsw, epsU, burnin, maxepoch,param_seed); toc();
testRMSE=Array(Float64,maxepoch)
finalpred=zeros(Ntest)
numbatches=int(ceil(Ntrain/m))
for epoch=1:maxepoch
    testpred=pred(w_store[:,epoch*numbatches],U_store[:,:,:,epoch*numbatches],I,phitest)
	if (maxepoch-epoch)<50
		finalpred+=testpred
	end
    testRMSE[epoch]=ytrainStd*norm(ytest-testpred)/sqrt(Ntest)
end
plot(testRMSE)
finalpred/=50
println("vanilla SGLD RMSE over last 50 epochs=",ytrainStd*norm(ytest-finalpred)/sqrt(Ntest))
#end
=#
testpred=pred(w,U,I,phitest);
println(ytrainStd*norm(ytest-testpred)/sqrt(Ntest))
