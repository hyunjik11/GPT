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
@everywhere C=7;
@everywhere Ntrain=1300;
@everywhere Ntest=N-Ntrain;
@everywhere seed=17;
@everywhere length_scale=1.4332;
@everywhere sigma_RBF=1;
@everywhere signal_var=0.2299^2;
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

function neglogjointlkhd(theta::Array,length_scale,sigma_RBF::Real,signal_var::Real)
    phi=featureNotensor(Xtrain,n,length_scale,sigma_RBF,seed)
    phi_theta=phi'*theta
    exp_phi_theta=exp(phi'*theta)
    L=0;
    for i=1:Ntrain
        L+=log(sum(exp_phi_theta[i,:]))-phi_theta[i,ytrain[i]]
    end
    L+=sum(abs2(theta))/2
    return L 
end

function gradneglogjointlkhd(theta::Array,length_scale,sigma_RBF::Real,signal_var::Real)
	phi=featureNotensor(Xtrain,n,length_scale,sigma_RBF,seed)
	res=ytrain-phi'*theta
	gradtheta=theta-phi*res/signal_var
	gradfeature=gradfeatureNotensor(Xtrain,length_scale,sigma_RBF,seed,phi)
	gradlength_scale=-theta'*gradfeature[1]*res
	gradsigma_RBF=-theta'*gradfeature[2]*res
	gradsignal_var=Ntrain/(2*signal_var)-sum(res.^2)/(2*signal_var^2)
	return [gradtheta,gradlength_scale,gradsigma_RBF,gradsignal_var]
end
