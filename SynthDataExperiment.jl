@everywhere using GPT_SGLD
@everywhere using PyPlot
@everywhere using HDF5
#@everywhere using Iterators

@everywhere file="TensorSynthData5D100N.h5";
@everywhere X=h5read(file,"X");
@everywhere y=h5read(file,"y3"); @everywhere signal_var=1e-3;
@everywhere w=h5read(file,"w");
@everywhere U=h5read(file,"U");
@everywhere I=h5read(file,"I");
@everywhere phi=h5read(file,"phi");
@everywhere length_scale=h5read(file,"length_scale");
@everywhere N=size(X,1);
@everywhere D=5;
@everywhere Ntrain=int(N/2);
@everywhere Ntest=N-Ntrain;
@everywhere seed=15;
@everywhere srand(seed);
@everywhere sigma_RBF=1;
@everywhere Xtrain = X[1:Ntrain,:];
@everywhere ytrain = y[1:Ntrain];
@everywhere Xtest = X[Ntrain+1:N,:];
@everywhere ytest = y[Ntrain+1:N];
@everywhere burnin=0;
@everywhere maxepoch=300;
@everywhere Q=32;
@everywhere m=10;
@everywhere r=2;
@everywhere n=5;
@everywhere Z=randn(n,D);
@everywhere b=2*pi*rand(n);
@everywhere scale=sqrt(n/(Q^(1/D)));
#@everywhere ytrainMean=mean(ytrain); ytrainStd=std(ytrain); ytrain=(ytrain-ytrainMean)/ytrainStd; ytest=(ytest-ytrainMean)/ytrainStd; phitrain=featureNotensor(Xtrain,length_scale,sigma_RBF,Z,b); phitest=featureNotensor(Xtest,length_scale,sigma_RBF,Z,b);
#@everywhere phitrain=feature(Xtrain,n,length_scale,sigma_RBF,seed,scale); @everywhere phitest=feature(Xtest,n,length_scale,sigma_RBF,seed,scale);
@everywhere phitrain=phi[:,:,1:Ntrain]; @everywhere phitest=phi[:,:,Ntrain+1:end];
@everywhere epsw=1e-5; 
@everywhere epsU=1e-8; 
@everywhere epsilon=1e-7; #100:3e-6,500:1e-6,1000:5e-4 2000:1e-5 4000:1e-7
@everywhere decay_rate=0;
@everywhere alpha=0.99;
@everywhere L=30;
@everywhere param_seed=1;
for param_seed=1:10
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
plot(testRMSE,color="magenta",label=string("n=",n))
finalpred/=50
println("n=",n,";epsilon=",epsilon,";seed=",seed,";vanilla SGLD RMSE over last 50 epochs=",ytrainStd*norm(ytest-finalpred)/sqrt(Ntest))
end

@everywhere t=Iterators.product(1:10,1:10)
@everywhere myt=Array(Any,100);
@everywhere it=1;
@everywhere for prod in t
	myt[it]=prod;
        it+=1;
        end
@sync @parallel for  Tuple in myt
i,j=Tuple;
epsw=float(string(i,"e-6")); epsU=float(string(j,"e-9"));
=#
#for param_seed=1:5
tic(); w_store,U_store=GPTregression(phitrain, ytrain, signal_var, I, r, Q, m, epsw, epsU, burnin, maxepoch,param_seed); toc();
testRMSE=Array(Float64,maxepoch)
finalpred=zeros(Ntest)
numbatches=int(ceil(Ntrain/m))
for epoch=1:maxepoch
    testpred=pred(w_store[:,epoch*numbatches],U_store[:,:,:,epoch*numbatches],I,phitest)
	if (maxepoch-epoch)<50
		finalpred+=testpred
	end
    testRMSE[epoch]=norm(ytest-testpred)/sqrt(Ntest)
end
plot(testRMSE)
finalpred/=50
println("n=",n,";epsw=",epsw,";epsU=",epsU,";vanilla SGLD RMSE over last 50 epochs=",norm(ytest-finalpred)/sqrt(Ntest))
    #end
    end
#testpred=pred(w,U,I,phitest)
#println(norm(ytest-testpred)/sqrt(Ntest))

