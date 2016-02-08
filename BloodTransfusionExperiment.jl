#include("GPkit.jl-master/src/GPkit.jl")

@everywhere using GPT_SGLD
@everywhere using DataFrames: readtable
#@everywhere using GPkit
#@everywhere using PyPlot
#@everywhere using Iterators: product

@everywhere data=readtable("transfusion.data",header=true);
@everywhere data=convert(Array,data);
@everywhere data=float(data);
@everywhere N=size(data,1);
@everywhere D=4;
@everywhere C=2;
@everywhere Ntrain=500;
@everywhere Ntest=N-Ntrain;
@everywhere seed=17;
@everywhere length_scale=1.4332;
@everywhere sigma_RBF=1;
@everywhere Xtrain = data[1:Ntrain,1:D];
@everywhere ytrain = int(data[1:Ntrain,D+1])+1;
@everywhere XtrainMean=mean(Xtrain,1); 
@everywhere XtrainStd=zeros(1,D);
@everywhere for i=1:D
	    XtrainStd[1,i]=std(Xtrain[:,i]);
	    end
@everywhere Xtrain = datawhitening(Xtrain);
@everywhere Xtest = (data[Ntrain+1:Ntrain+Ntest,1:D]-repmat(XtrainMean,Ntest,1))./repmat(XtrainStd,Ntest,1);
@everywhere ytest = int(data[Ntrain+1:Ntrain+Ntest,D+1])+1
@everywhere burnin=0;
@everywhere maxepoch=200;
@everywhere Q=200;
@everywhere m=50;
@everywhere r=10;
@everywhere n=300;
@everywhere I=samplenz(r,D,Q,seed);
@everywhere scale=sqrt(n/(Q^(1/D)));
@everywhere phitrain=featureNotensor(Xtrain,n,length_scale,sigma_RBF,seed);
@everywhere phitest=featureNotensor(Xtest,n,length_scale,sigma_RBF,seed);
@everywhere epsw=1e-4; 
@everywhere epsU=1e-7;
@everywhere epsilon=1e-8;
@everywhere alpha=0.99;
@everywhere L=30;
@everywhere param_seed=234;

# computes log(sum(exp(x))) in a robust manner
function logsumexp(x::Array)
    a=maximum(x);
    return a+log(sum(exp(x-a)))
end

#input theta_vector is a vector of length n*C - converted to n by C array within function
function neglogjointlkhd(theta_vec::Vector,hyperparams::Vector)
	if length(hyperparams)==2 # if length_scale is a scalar	
		length_scale=hyperparams[1]
	else length_scale=hyperparams[1:end-1]
	end
	sigma_RBF=hyperparams[end]
	theta=reshape(theta_vec,n,C) # n by C matrix
    phi=featureNotensor(Xtrain,n,length_scale,sigma_RBF,seed) # n by Ntrain matrix
    phi_theta=theta'*phi # C by Ntrain matrix
    L=0;
    for i=1:Ntrain
        L+=logsumexp(phi_theta[:,i])-phi_theta[ytrain[i],i]
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
	    phi_theta=theta'*phi	# C by Ntrain matrix
	    gradtheta=zeros(n,C)
	    for c=1:C
		for i=1:Ntrain
		    gradtheta[:,c]+=exp(phi_theta[c,i]-logsumexp(phi_theta[:,i]))*phi[:,i]
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
                gradlength_scale+=sum(exp(phi_theta[:,i]-logsumexp(phi_theta[:,i])).*(theta'*gradfeature[1][:,i]))-sum(theta[:,ytrain[i]].*gradfeature[1][:,i])
		gradsigma_RBF+=sum(exp(phi_theta[:,i]-logsumexp(phi_theta[:,i])).*(theta'*gradfeature[2][:,i]))-sum(theta[:,ytrain[i]].*gradfeature[2][:,i])
	    end
	else # length_scale is a vector - varying length scales across input dimensions
        length_scale=hyperparams[1:end-1] 
	    phi=featureNotensor(Xtrain,n,length_scale,sigma_RBF,seed) # n by Ntrain matrix
	    phi_theta=theta'*phi	# C by Ntrain matrix
	    gradtheta=zeros(n,C)
            for c=1:C
                for i=1:Ntrain
		    gradtheta[:,c]+=exp(phi_theta[c,i]-logsumexp(phi_theta[:,i]))*phi[:,i]
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
                for k=1:D
                    gradlength_scale[k]+=sum(exp(phi_theta[:,i]-logsumexp(phi_theta[:,i])).*(theta'*gradfeature[1][:,i,k]))-sum(theta[:,ytrain[i]].*gradfeature[1][:,i,k])
                end
                gradsigma_RBF+=sum(exp(phi_theta[:,i]-logsumexp(phi_theta[:,i])).*(theta'*gradfeature[2][:,i]))-sum(theta[:,ytrain[i]].*gradfeature[2][:,i])
	    end
	end
	return [vec(gradtheta),gradlength_scale,gradsigma_RBF]
end

function testng(init_theta::Vector,init_hyperparams::Vector,
neglogjointlkhd::Function,gradneglogjointlkhd::Function;epsilont::Real=1e-2,epsilonh::Real=1e-5,num_sg_iter::Integer=10,num_sgld_iter::Integer=10,alpha::Real=0.9)
	# neglogjointlkhd should be -log p(y,theta;hyperparams), a function with 
	# input theta,length_scale,sigma_RBF,signal_var and scalar output
	# gradneglogjointlkhd should be the gradient of neglogjointlkhd wrt theta and hyperparams with
	# input theta,length_scale,sigma_RBF,signal_var and vector output of length equal to length(theta)+3
	
	nc=length(init_theta);
	Lh=length(init_hyperparams);

	# initialise theta and loghyperparams
	theta=init_theta; loghyperparams=log(init_hyperparams)

	# define f which corresponds to neglogjointlkhd and gradneglogjointlkhd but with inputs loghyperparams instead of hyperparams (for optimisation's sake) and g its gradient - compute using chain rule
	f(theta,loghyperparameters)=neglogjointlkhd(theta,exp(loghyperparameters));
	g(theta,loghyperparameters)=gradneglogjointlkhd(theta,exp(loghyperparameters)).*[ones(nc),exp(loghyperparameters)];

	# stochastic EM 
	gtheta=zeros(nc); # moving average of theta gradients
	gh=zeros(Lh); # moving average of loghyperparam gradients
	# initialise statistic for diagnosing convergence
	absdiff=1; # |x - x'|
	iter=1;
	while absdiff>1e-7
		println("iteration ",iter)
		# E step - sample theta from posterior using SGLD with RMS_prop - but then need to decide on step size
		#=		
		for i=1:num_sgld_iter
			gradtheta=g(theta,loghyperparams)[1:nc];
			gtheta=alpha*gtheta+(1-alpha)*(gradtheta.^2);
			epstheta=epsilont./(sqrt(gtheta)+1e-5);
			println("epstheta norm=",norm(epstheta));
			println("theta gradient norm=",norm(gradtheta));
			theta-=epstheta.*gradtheta/2+sqrt(epstheta).*randn(nc)
			println("theta norm=",norm(theta))
		end
		=#
		for i=1:num_sgld_iter
			gradtheta=g(theta,loghyperparams)[1:nc];
			theta-=epsilont*gradtheta/2+sqrt(epsilont)*randn(nc)
			println("theta gradient norm=",norm(gradtheta));
			println("theta norm=",norm(theta))
		end
		# M step - maximisie joint log likelihood wrt hyperparams using no_cg_iter steps of cg/gd
		
		println("function_value_before:",f(theta,loghyperparams))
		new_loghyperparams=loghyperparams;
		for i=1:num_sg_iter
			loghypergrad=g(theta,new_loghyperparams)[end-Lh+1:end];
			gh=alpha*gh+(1-alpha)*(loghypergrad.^2)
			epsh=epsilonh./(sqrt(gh)+1e-5);
			println("epsh norm=",norm(epsh));
			new_loghyperparams-=epsh.*loghypergrad
			println("loghyperparam gradients:",loghypergrad);
		end
		println("function_value_after:",f(theta,new_loghyperparams))
		println("hyperparams:",exp(new_loghyperparams))

		#update convergence statistics
		absdiff=norm(exp(loghyperparams)-exp(new_loghyperparams));
		println("|x-x'|: ",absdiff)
		println()

		#update hyperparams
		loghyperparams=new_loghyperparams
		iter+=1
	end
	
	return exp(loghyperparams)

end	

init_length_scale=[2.1594,1.3297,1.3283,2.1715];
init_sigma_RBF=1.2459;
hyperparams=[init_length_scale,init_sigma_RBF];
init_theta=randn(n*C);
theta=init_theta;
epsilont=1e-2;
nlp=Array(Float64,Ntest);
prediction=Array(Integer,Ntest);
gtheta=zeros(n*C);
for i=1:1000
    gradtheta=gradneglogjointlkhd(theta,hyperparams)[1:n*C];
    gtheta=alpha*gtheta+(1-alpha)*(gradtheta.^2);
    epstheta=epsilont./(sqrt(gtheta)+1e-5);
    #println("epstheta norm=",norm(epstheta));
    theta-=epstheta.*gradtheta/2+sqrt(epstheta).*randn(n*C)
    #println("theta norm=",norm(theta))
    #theta-=epsilon*gradneglogjointlkhd(theta,[init_length_scale,init_sigma_RBF])[1:n*C]/2#+sqrt(epsilon)*randn(n*C)
    fhat_test=phitest'*reshape(theta,n,C);
    for j=1:Ntest
	prediction[j]=indmax(fhat_test[j,:])
	nlp[j]=logsumexp(fhat_test[j,:])-fhat_test[j,ytest[j]]
    end
    if i%10==0
        println("prop_missed=",1-sum(prediction.==ytest)/Ntest," mean_nlp=",mean(nlp)," theta norm=",norm(theta)," theta gradient norm=",norm(gradtheta))
    end
end
#testng(init_theta,[init_length_scale,init_sigma_RBF],neglogjointlkhd,gradneglogjointlkhd,num_sgld_iter=1000)



