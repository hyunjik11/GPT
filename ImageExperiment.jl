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
@everywhere n=50;
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

#input theta_vector is a vector of length n*C - converted to n by C array within function
function neglogjointlkhd(theta_vec::Vector,hyperparams::Vector)
	if length(hyperparams)==2 # if length_scale is a scalar	
		length_scale=hyperparams[1]
	else length_scale=hyperparams[1:end-1]
	end
	sigma_RBF=hyperparams[end]
	theta=reshape(theta_vec,n,C) # n by C matrix
    phi=featureNotensor(Xtrain,n,length_scale,sigma_RBF,seed) # n by Ntrain matrix
    phi_theta=phi'*theta # Ntrain by C matrix
    exp_phi_theta=exp(phi_theta) 
    L=0;
    for i=1:Ntrain
        L+=log(sum(exp_phi_theta[i,:]))-phi_theta[i,ytrain[i]]
    end
    L+=sum(abs2(theta))/2
    return L 
end

function gradneglogjointlkhd(theta_vec::Vector,hyperparams::Vector)
	theta=reshape(theta_vec,n,C) # n by C matrix
	sigma_RBF=hyperparams[end]	

	if length(hyperparams)==2 # if length_scale is a scalar
		length_scale=hyperparams[1]	
		phi=featureNotensor(Xtrain,n,length_scale,sigma_RBF,seed) # n by Ntrain matrix
		exp_phi_theta=exp(phi'*theta)	# Ntrain by C matrix
		sum_exp_phi_theta=sum(exp_phi_theta,2) # Vector length Ntrain
		gradtheta=zeros(n,C)
		for c=1:C
			for i=1:Ntrain
				gradtheta[:,c]+=exp_phi_theta[i,c]*phi[:,i]/sum_exp_phi_theta[i]
			end
		end
		for i=1:Ntrain
			gradtheta[:,ytrain[i]]-=phi[:,i]
		end
		gradtheta+=theta

		gradfeature=gradfeatureNotensor(Xtrain,n,length_scale,sigma_RBF,seed)
		gradlength_scale=0;
		gradsigma_RBF=0;
		for i=1:Ntrain
			gradlength_scale+=sum(vec(exp_phi_theta[i,:]).*(theta'*gradfeature[1][:,i]))/sum_exp_phi_theta[i]-sum(theta[:,ytrain[i]].*gradfeature[1][:,i])
			gradsigma_RBF+=sum(vec(exp_phi_theta[i,:]).*(theta'*gradfeature[2][:,i]))/sum_exp_phi_theta[i]-sum(theta[:,ytrain[i]].*gradfeature[2][:,i])
		end
	else # length_scale is a vector - varying length scales across input dimensions
		length_scale=hyperparams[1:end-1] 
		phi=featureNotensor(Xtrain,n,length_scale,sigma_RBF,seed) # n by Ntrain matrix
		exp_phi_theta=exp(phi'*theta)	# Ntrain by C matrix
		sum_exp_phi_theta=sum(exp_phi_theta,2) # Vector length Ntrain
		gradtheta=zeros(n,C)
		for c=1:C
			for i=1:Ntrain
				gradtheta[:,c]+=exp_phi_theta[i,c]*phi[:,i]/sum_exp_phi_theta[i]
			end
		end
		for i=1:Ntrain
			gradtheta[:,ytrain[i]]-=phi[:,i]
		end
		gradtheta+=theta
	
		gradfeature=gradfeatureNotensor(Xtrain,n,length_scale,sigma_RBF,seed)
		gradlength_scale=zeros(D);
		gradsigma_RBF=0;
		for i=1:Ntrain
			dphi_xibydl=squeeze(gradfeature[1][:,i,:],2)
			gradlength_scale+=vec(exp_phi_theta[i,:]*theta'*dphi_xibydl/sum_exp_phi_theta[i]-theta[:,ytrain[i]]'*dphi_xibydl)
			gradsigma_RBF+=sum(vec(exp_phi_theta[i,:]).*(theta'*gradfeature[2][:,i]))/sum_exp_phi_theta[i]-sum(theta[:,ytrain[i]].*gradfeature[2][:,i])
		end
	end
	return [vec(gradtheta),gradlength_scale,gradsigma_RBF]
end

init_length_scale=1;
init_sigma_RBF=1;
init_theta=randn(n*C);
GPNT_hyperparameters_ng(init_theta,[init_length_scale,init_sigma_RBF],
neglogjointlkhd,gradneglogjointlkhd)




