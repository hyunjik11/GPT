#include("GPkit.jl-master/src/GPkit.jl")

@everywhere using GPT_SGLD
@everywhere using DataFrames
#@everywhere using Distributions
#@everywhere using GPkit
#@everywhere using PyPlot
#@everywhere using Iterators

@everywhere data=DataFrames.readtable("Folds5x2_pp.csv", header = true);
@everywhere data = convert(Array,data);
@everywhere N=size(data,1);
@everywhere D=4;
@everywhere Ntrain=500;
@everywhere Ntest=N-Ntrain;
@everywhere seed=18;
@everywhere length_scale=1.4332;
@everywhere sigma_RBF=1;
#@everywhere length_scale=1+0.2*randn(1)[1];
#@everywhere sigma_RBF=1+0.2*randn(1)[1];
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
@everywhere maxepoch=100;
@everywhere Q=200;
@everywhere m=10;
@everywhere r=20;
@everywhere n=320;
for seed=1:10
@everywhere srand(seed)
@everywhere Z=randn(n,D);
@everywhere b=2*pi*rand(n,D);
@everywhere I=samplenz(r,D,Q);
@everywhere phi_scale=sqrt(n/(Q^(1/D)));
@everywhere phitrain=featureNotensor(Xtrain,length_scale,sigma_RBF,Z,b);
@everywhere phitest=featureNotensor(Xtest,length_scale,sigma_RBF,Z,b);
@everywhere epsw=1e-4; 
@everywhere epsU=1e-7;
@everywhere epsilon=1e-4;
@everywhere alpha=0.99;
@everywhere L=30;
@everywhere param_seed=234;
#tic();w_store,U_store,accept_prob=GPT_GMC(phitrain,ytrain,sigma,I,r,Q,epsw,epsU,burnin,maxepoch,L,param_seed);toc()

randfeature(hyperparams::Vector)=featureNotensor(Xtrain,hyperparams[1:D],hyperparams[D+1],Z,b)
gradfeature(hyperparams::Vector)=gradfeatureNotensor(Xtrain,hyperparams[1:D],hyperparams[D+1],Z,b)
nlogmarginal(hyperparams::Vector)=GPNT_nlogmarginal(ytrain,n,hyperparams,randfeature)
gradnlogmarginal(hyperparams::Vector)=GPNT_gradnlogmarginal(ytrain,n,hyperparams,randfeature,gradfeature)
#test(nlogmarginal,gradnlogmarginal,1+0.2*randn(3),epsilon)

Lh=D+2; #number of hyperparams
#lbounds=[0.,0.,0.001] #lower bound on hyperparams
#GPNT_hyperparameters(nlogmarginal,gradnlogmarginal,1+0.2*randn(Lh),lbounds)
myhyperparams,fmin=GPNT_hyperparameters(nlogmarginal,gradnlogmarginal,[ones(Lh-1),0.2^2],0.001*ones(Lh));
length_scale=myhyperparams[1:D]; sigma_RBF=myhyperparams[D+1]; signal_var=myhyperparams[D+2];
println("-l=$fmin;length_scale=$length_scale;sigma_RBF=$sigma_RBF;signal_var=$signal_var");
end
#=
function test(nlogmarginal::Function,gradnlogmarginal::Function,init_hyperparams::Vector,epsilon::Real)
nlm(loghyperparams::Vector)=nlogmarginal(exp(loghyperparams)); # exp needed to enable unconstrained optimisation, since hyperparams must be positive
gnlm(loghyperparams::Vector)=gradnlogmarginal(exp(loghyperparams)).*exp(loghyperparams)
loghyperparams=log(init_hyperparams)
for i=1:100
	println("nlogmarginal=",nlm(loghyperparams)," hyperparams=",exp(loghyperparams))
	loghyperparams-=epsilon*gnlm(loghyperparams)	
end
return exp(loghyperparams)
end
=#
#=
nll=GPNT_logmarginal(Xtrain,ytrain,length_scale,sigma_RBF,signal_var,Z,b)
println("nll=",nll," Z[1,1]=",Z[1,1])

cov=CovSEiso(length_scale,sigma_RBF);
lik=LikGauss(signal_var);
gp=GPmodel(InfExact(), cov, lik, MeanZero(), Xtrain, ytrain);
tic(); (post,nlZ,dnlZ)=inference(gp, with_dnlz=false); toc()
tic(); (ymu,ys2,fmu,fs2,lp)=prediction(gp, post, Xtest); toc()
println(ytrainStd*norm(ytest-ymu)/sqrt(Ntest))

function neglogjointlkhd(theta::Vector,hyperparams::Vector)
	length_scale,sigma_RBF,signal_var=hyperparams
	phi=featureNotensor(Xtrain,n,length_scale,sigma_RBF,seed)
	res=ytrain-phi'*theta
	return sum(res.^2)/(2*signal_var)+Ntrain*log(signal_var)/2+sum(theta.^2)/2
end

function gradneglogjointlkhd(theta::Vector,hyperparams::Vector)
	length_scale,sigma_RBF,signal_var=hyperparams
	phi=featureNotensor(Xtrain,n,length_scale,sigma_RBF,seed)
	res=ytrain-phi'*theta
	gradtheta=theta-phi*res/signal_var
	gradfeature=gradfeatureNotensor(Xtrain,n,length_scale,sigma_RBF,seed)
	gradlength_scale=-theta'*gradfeature[1]*res
	gradsigma_RBF=-theta'*gradfeature[2]*res
	gradsignal_var=Ntrain/(2*signal_var)-sum(res.^2)/(2*signal_var^2)
	return [gradtheta,gradlength_scale,gradsigma_RBF,gradsignal_var]
end

function neglogjointlkhd(theta::Vector,length_scale::Real,sigma_RBF::Real,signal_var::Real)
	phi=featureNotensor(Xtrain,n,length_scale,sigma_RBF,seed)
	lkhd=MvNormal(phi'theta,sqrt(signal_var))
	theta_prior=MvNormal(n,1)
	return -log(pdf(lkhd,ytrain))-log(pdf(theta_prior,theta))
end

neglogjointlkhd2(params::Vector)=neglogjointlkhd(params[1:n],params[end-2],params[end-1],params[end])
gradneglogjointlkhd2=ForwardDiff.gradient(neglogjointlkhd2)
gradneglogjointlkhd(theta::Vector,length_scale::Real,sigma_RBF::Real,signal_var::Real)=gradneglogjointlkhd2([theta,length_scale,sigma_RBF,signal_var])


init_theta=randn(n);
init_length_scale=1.0
init_sigma_RBF=1.0
init_signal_var=0.04

myhyperparams=GPNT_hyperparameters_ng(init_theta,[init_length_scale,init_sigma_RBF,init_signal_var],neglogjointlkhd,gradneglogjointlkhd)
=#
#=
tic(); w_store,U_store=GPT_SGDE(phitrain, ytrain, signal_var, I, r, Q, m, epsw, epsU, burnin, maxepoch); toc();
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

tic(); w_store,U_store=GPT_SGLDERM_RMSprop(phitrain, ytrain, signal_var, I, r, Q, m, epsilon, alpha, burnin, maxepoch); toc();
testRMSE2=Array(Float64,maxepoch)
finalpred=zeros(Ntest)
numbatches=int(ceil(Ntrain/m))
for epoch=1:maxepoch
    testpred=pred(w_store[:,epoch*numbatches],U_store[:,:,:,epoch*numbatches],I,phitest)
	if (maxepoch-epoch)<50
		finalpred+=testpred
	end
    testRMSE2[epoch]=ytrainStd*norm(ytest-testpred)/sqrt(Ntest)
end
finalpred/=50
println("RMSprop SGLD RMSE over last 100 epochs=",ytrainStd*norm(ytest-finalpred)/sqrt(Ntest))

plot(testRMSE,label="vanilla SGLD"); title("PowerPlant testRMSE. SGLD vs SGLD_RMSprop");
plot(testRMSE2,label="RMSprop SGLD"); legend(); savefig("PowerPlantRMSEwithSGLD_RMSprop");
=#


#=
    trainRMSE=SharedArray(Float64,maxepoch);
    testRMSE=SharedArray(Float64,maxepoch);
    trainfhat=SharedArray(Float64,Ntrain,10);
    testfhat=SharedArray(Float64,N-Ntrain,10);
    @sync @parallel for i=1:maxepoch
	fhattrain=pred(w_store[:,i],U_store[:,:,:,i],I,phitrain);
	trainRMSE[i]=ytrainStd*norm(ytrain-fhattrain)/sqrt(Ntrain);
	fhattest=pred(w_store[:,i],U_store[:,:,:,i],I,phitest);
	testRMSE[i]=ytrainStd*norm(ytest-fhattest)/sqrt(N-Ntrain);
	if i>maxepoch-10
		trainfhat[:,i-(maxepoch-10)]=fhattrain
		testfhat[:,i-(maxepoch-10)]=fhattest
	end
    end
println(" trainRMSE=",ytrainStd*norm(ytrain-mean(trainfhat,2))/sqrt(Ntrain),"epsw=",epsw," epsU=",epsU," L=",L," maxepoch=",maxepoch)
println(" testRMSE=",ytrainStd*norm(ytest-mean(testfhat,2))/sqrt(N-Ntrain),"epsw=",epsw," epsU=",epsU," L=",L," maxepoch=",maxepoch)
end
=#

#=
@everywhere T=maxepoch*int(floor(Ntrain/m));
trainRMSE=SharedArray(Float64,T);
testRMSE=SharedArray(Float64,T);
trainfhat=SharedArray(Float64,Ntrain,100);
testfhat=SharedArray(Float64,N-Ntrain,100);
@sync @parallel for i=1:T
	fhattrain=pred(w_store[:,i],U,I,phitrain);
	trainRMSE[i]=ytrainStd*norm(ytrain-fhattrain)/sqrt(Ntrain);
	fhattest=pred(w_store[:,i],U,I,phitest);
	testRMSE[i]=ytrainStd*norm(ytest-fhattest)/sqrt(N-Ntrain);
	if i>T-100
		trainfhat[:,i-(T-100)]=fhattrain
		testfhat[:,i-(T-100)]=fhattest
	end
end
println("epsw=",epsw," epsU=",epsU,"trainRMSE=",ytrainStd*norm(ytrain-mean(trainfhat,2))/sqrt(Ntrain))
println("epsw=",epsw," epsU=",epsU,"testRMSE=",ytrainStd*norm(ytest-mean(testfhat,2))/sqrt(N-Ntrain))
#plot(trainRMSE)
#plot(testRMSE)
#tic();myRMSEidx,temp=RMSE(w_store,U_store,I,phitest,ytest);toc();
=#

#=
@everywhere t=Iterators.product(5:5:50,3:5,5:7)
@everywhere myt=Array(Any,90);
@everywhere it=1;
@everywhere for prod in t
	myt[it]=prod;
        it+=1;
        end
#myRMSE=SharedArray(Float64,70);
@sync @parallel for  Tuple in myt
    r,i=Tuple;
    epsw=float(string("1e-",i)); #epsU=float(string("1e-",j));
    I=samplenz(r,D,Q,seed);
    #idx=int(3*(j-70)/5+i-14);
    w_store,U=GPT_SGLDERMw(phitrain,ytrain,sigma,I,r,Q,m,epsw,burnin,maxepoch);
    
    testRMSE=Array(Float64,maxepoch)
    numbatches=int(ceil(Ntrain/m))
    for epoch=1:maxepoch
        testpred=pred(w_store[:,epoch*numbatches],U,I,phitest)
        testRMSE[epoch]=ytrainStd*norm(ytest-testpred)/sqrt(Ntest)
    end
    
	#println("RMSE=",myRMSE,";seed=",seed,";sigma=",sigma,";length_scale=",length_scale,";n=",n,";r=",r,";Q=",Q,";m=",m,";epsw=",epsw,";epsU=",epsU,";burnin=",burnin,";maxepoch=",maxepoch);
println("r=",r,";minRMSE=",minimum(testRMSE),";minepoch=",indmin(testRMSE),";epsw=",epsw,";burnin=",burnin,";maxepoch=",maxepoch);
end
=#

#=
numI=50;
meanfhat=Array(Float64,N-Ntrain,numI);
for iseed=1:numI
	I=samplenz(r,D,Q,iseed);
	w_store,U_store=GPT_SGLDERM(phitrain,ytrain,sigma,I,r,Q,m,epsw,epsU,burnin,maxepoch);
	temp,meanfhat[:,iseed]=RMSE(w_store,U_store,I,phitest,ytest);
	println(iseed," iteration out of ",numI," done");
end
meanfhatfinal=mean(meanfhat,2);
println("RMSE=",norm(ytest-meanfhatfinal)/sqrt(N-Ntrain))
=#

if 1==0 #storing variables to h5 file
using HDF5
c=h5open("SynthData1000.h5","w") do file
	write(file,"Xtrain",sdata(Xtrain));
	write(file,"XtrainMean",sdata(XtrainMean));
end
end

if 1==0 #reading variables from h5 file
using HDF5
file="10000SynthData.h5";
Xtrain=h5read(file,"Xtrain");
XtrainMean=h5read(file,"XtrainMean");
end

if 1==0 #writing stdout to file
    	outfile=open(filename,"a") #append to file
	println(outfile,"RMSE=",ytrainStd*predRMSE,";seed=",seed,";sigma=",sigma, 		";length_scale=",length_scale,";n=",n,";r=",r,";Q=",Q,";m=",m,";epsw=",epsw, 		";epsU=",epsU,";burnin=",burnin,";maxepoch=",maxepoch);
	close(outfile)
end



