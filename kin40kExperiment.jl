#include("GPkit.jl-master/src/GPkit.jl")

@everywhere using GPT_SGLD
@everywhere using DataFrames
#@everywhere using GPkit
@everywhere using HDF5
#@everywhere using PyPlot
#@everywhere using Iterators


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
@everywhere maxepoch=200;
@everywhere Q=200;
@everywhere m=50;
@everywhere r=20;
@everywhere n=150;
@everywhere I=samplenz(r,D,Q,seed);
@everywhere scale=sqrt(n/(Q^(1/D)));
@everywhere phitrain=feature(Xtrain,n,length_scale,sigma_RBF,seed,scale);
@everywhere phitest=feature(Xtest,n,length_scale,sigma_RBF,seed,scale);
#@everywhere epsilon=7*1e-10;
#@everywhere alpha=0.99;
@everywhere epsw=1e-4;
@everywhere epsU=1e-7;
@everywhere numbatches=int(ceil(Ntrain/m))

cov=CovSEiso(length_scale,sigma_RBF);
lik=LikGauss(signal_var);
gp=GPmodel(InfExact(), cov, lik, MeanZero(), Xtrain, ytrain);
tic(); (post,nlZ,dnlZ)=inference(gp, with_dnlz=false); toc()
tic(); (ymu,ys2,fmu,fs2,lp)=prediction(gp, post, Xtest); toc()
println(ytrainStd*norm(ytest-ymu)/sqrt(Ntest))

#=

#(optf,optx,ret) = optinf(gp, 200, algo=:LD_LBFGS, with_dnlz=true); # optf has new hypers


tic(); w_store,U_store=GPT_SGLDERM(phitrain,ytrain,signal_var,I,r,Q,m,epsw,epsU,burnin,maxepoch); toc();


w_store_thin=Array(Float64,Q,maxepoch);
U_store_thin=Array(Float64,n,r,D,maxepoch);
for epoch=1:maxepoch
	w_store_thin[:,epoch]=w_store[:,epoch*numbatches]
	U_store_thin[:,:,:,epoch]=U_store[:,:,:,epoch*numbatches]
end


c=h5open("wU_store_kin40k.h5","w") do file
	write(file,"w_store",w_store_thin);
	write(file,"U_store",U_store_thin);
end



@everywhere file="wU_store_kin40k.h5";
@everywhere w_store=h5read(file,"w_store");
@everywhere U_store=h5read(file,"U_store");

testRMSE=SharedArray(Float64,maxepoch);
testpred=SharedArray(Float64,Ntest,maxepoch);
@sync @parallel for epoch=1:maxepoch
	testpredtemp=pred(w_store[:,epoch],U_store[:,:,:,epoch],I,phitest);
    testpred[:,epoch]=testpredtemp
    testRMSE[epoch]=ytrainStd*norm(ytest-testpredtemp)/sqrt(Ntest)
	println("epoch ",epoch," done")
end
=#
#GPNT_hyperparameters(Xtrain,ytrain,n,length_scale,sigma_RBF,signal_var,seed)
#=
cov=CovSEiso(length_scale,sigma_RBF);
lik=LikGauss(signal_var);
gp=GPmodel(InfExact(), cov, lik, MeanZero(), Xtrain, ytrain);
tic(); (post,nlZ,dnlZ)=inference(gp, with_dnlz=false); toc()
tic(); (ymu,ys2,fmu,fs2,lp)=prediction(gp, post, Xtest); toc()
println(ytrainStd*norm(ytest-ymu)/sqrt(Ntest))

(optf,optx,ret) = optinf(gp, 200, algo=:LD_LBFGS, with_dnlz=true); # optf has new hypers

(ymu,ys2,fmu,fs2,lp)=prediction(gp, post, xs);  # optimised hypers were "left" in gp.covfn and gp.likfn


@everywhere t=Iterators.product(4:6,6:9)
@everywhere myt=Array(Any,12);
@everywhere it=1;
@everywhere for prod in t
	myt[it]=prod;
        it+=1;
        end
#myRMSE=SharedArray(Float64,70);
@sync @parallel for  Tuple in myt
    i,j=Tuple;
    epsw=float(string("1e-",i)); epsU=float(string("1e-",j));
    I=samplenz(r,D,Q,seed);
    #idx=int(3*(j-70)/5+i-14);
    w_store,U_store=GPT_SGLDERM(phitrain,ytrain,signal_var,I,r,Q,m,epsw,epsU,burnin,maxepoch);
    testRMSE=Array(Float64,maxepoch)
    numbatches=int(ceil(Ntrain/m))
    for epoch=1:maxepoch
        testpred=pred(w_store[:,epoch*numbatches],U_store[:,:,:,epoch*numbatches],I,phitest)
        testRMSE[epoch]=ytrainStd*norm(ytest-testpred)/sqrt(Ntest)
    end    
	println("epsw=",epsw,";epsU=",epsU," last 10 testRMSEs:",testRMSE[maxepoch-9:maxepoch])	
	#plot(testRMSE); savefig(string("kin40ktestRMSE.epsw",epsw,"epsU",epsU));
	#println("minRMSE=",minimum(testRMSE),";minepoch=",indmin(testRMSE),";epsw=",epsw,";epsU=",epsU,";burnin=",burnin,";maxepoch=",maxepoch);
end
=#

