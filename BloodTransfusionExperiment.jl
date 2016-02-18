#include("GPkit.jl-master/src/GPkit.jl")

@everywhere using GPT_SGLD
@everywhere using DataFrames: readtable
@everywhere using Mamba
@everywhere using Optim
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
@everywhere seed=20;
@everywhere length_scale=[2.1594,1.3297,1.3283,2.1715];
#@everywhere length_scale=[1.3297,1.3283,2.1715];
@everywhere sigma_RBF=1.2459;
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
@everywhere m=500;
@everywhere r=10;
@everywhere n=5;
@everywhere srand(seed)
@everywhere I=samplenz(r,D,Q);
@everywhere Z=randn(n,D)
@everywhere b=2*pi*rand(n)
@everywhere scale=sqrt(n/(Q^(1/D)));
@everywhere phitrain=featureNotensor(Xtrain,length_scale,sigma_RBF,Z,b);
@everywhere phitest=featureNotensor(Xtest,length_scale,sigma_RBF,Z,b);
@everywhere epsw=1e-4; 
@everywhere epsU=1e-7;
@everywhere epsilon=1e-3;
@everywhere alpha=0.99;
@everywhere L=30;
@everywhere param_seed=234;

# computes log(sum(exp(x))) in a robust manner
function logsumexp(x::Array)
    a=maximum(x);
    return a+log(sum(exp(x-a)))
end

# input theta_vector is a vector of length n*C - converted to n by C array within function
# use the Z and b that was used to compute features
function neglogjointlkhd(theta_vec::Vector,hyperparams::Vector,Z::Array, b::Array)
	if length(hyperparams)==2 # if length_scale is a scalar	
		length_scale=hyperparams[1]
	else length_scale=hyperparams[1:end-1]
	end
	sigma_RBF=hyperparams[end]
	theta=reshape(theta_vec,n,C) # n by C matrix
	phi=featureNotensor(Xtrain,length_scale,sigma_RBF,Z,b) # n by Ntrain matrix
    phi_theta=theta'*phi # C by Ntrain matrix
    L=0;
    for i=1:Ntrain
        L+=logsumexp(phi_theta[:,i])-phi_theta[ytrain[i],i]
    end
    L+=sum(abs2(theta))/2
    return L 
end

# use the Z and b that was used to compute features
function gradneglogjointlkhd(theta_vec::Vector,hyperparams::Vector,Z::Array, b::Array)
	theta=reshape(theta_vec,n,C) # n by C matrix
	sigma_RBF=hyperparams[end]	
	if length(hyperparams)==2 # if length_scale is a scalar
	    length_scale=hyperparams[1]
	else length_scale=hyperparams[1:end-1]
	end
	phi=featureNotensor(Xtrain,length_scale,sigma_RBF,Z,b) # n by Ntrain matrix
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
        
	gradfeature=gradfeatureNotensor(Xtrain,length_scale,sigma_RBF,Z,b)
	gradsigma_RBF=0;
	if length(length_scale)==1
		gradlength_scale=0;
	    for i=1:Ntrain
                gradlength_scale+=sum(exp(phi_theta[:,i]-logsumexp(phi_theta[:,i])).*(theta'*gradfeature[1][:,i]))-sum(theta[:,ytrain[i]].*gradfeature[1][:,i])
		gradsigma_RBF+=sum(exp(phi_theta[:,i]-logsumexp(phi_theta[:,i])).*(theta'*gradfeature[2][:,i]))-sum(theta[:,ytrain[i]].*gradfeature[2][:,i])
	    end
	else # length_scale is a vector - varying length scales across input dimensions
        gradlength_scale=zeros(D);
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
neglogjointlkhd::Function,gradneglogjointlkhd::Function,Z::Array,b::Array;num_cg_iter::Integer=10)
	# neglogjointlkhd should be -log p(y,theta;hyperparams), a function with 
	# input theta,length_scale,sigma_RBF,signal_var and scalar output
	# gradneglogjointlkhd should be the gradient of neglogjointlkhd wrt theta and hyperparams with
	# input theta,length_scale,sigma_RBF,signal_var and vector output of length equal to length(theta)+3
	
	nc=length(init_theta);
	Lh=length(init_hyperparams);

	# initialise theta and loghyperparams
	theta=init_theta; loghyperparams=log(init_hyperparams)

	# define f which corresponds to neglogjointlkhd and gradneglogjointlkhd but with inputs loghyperparams instead of hyperparams (for optimisation's sake) and g its gradient - compute using chain rule
	f(theta,loghyperparameters)=neglogjointlkhd(theta,exp(loghyperparameters),Z,b);
	g(theta,loghyperparameters)=gradneglogjointlkhd(theta,exp(loghyperparameters),Z,b).*[ones(nc),exp(loghyperparameters)];

    model=Model(
	
	theta=Stochastic(2,
		@modelexpr(n,C, Distribution[Normal(0,1) for i in 1:n, j in 1:C])
	),


	fhat=Logical(2,
		@modelexpr(theta,phi,phi'*theta),
		false
	),
	
	p=Logical(2,
		@modelexpr(fhat,N,C,
		[exp(fhat[i,c]-maximum(fhat[i,:])-log(sum(exp(fhat[i,:]-maximum(fhat[i,:]))))) for c=1:C,i=1:N]),
		false
	),
	y=Stochastic(1,
		@modelexpr(p,N,
		Distribution[Categorical(p[:,i]) for i=1:N]
		),
	false
	)

    )
	# stochastic EM 
	gtheta=zeros(nc); # moving average of theta gradients
	gh=zeros(Lh); # moving average of loghyperparam gradients
	# initialise statistic for diagnosing convergence
	absdiff=1; # |x - x'|
	iter=1;
	while absdiff>1e-7
	    println("iteration ",iter)
	    # E step - sample theta from posterior using SGLD with RMS_prop - but then need to decide on step size
        length_scale=exp(loghyperparams[1:Lh-1]);
		sigma_RBF=exp(loghyperparams[Lh]);
	  	phitrain=featureNotensor(Xtrain,length_scale,sigma_RBF,Z,b);
        phitest=featureNotensor(Xtest,length_scale,sigma_RBF,Z,b);
        nlp=Array(Float64,Ntest);
        prediction=Array(Integer,Ntest);
        fhat_test=phitest'*reshape(theta,n,C);
        for j=1:Ntest
		    prediction[j]=indmax(fhat_test[j,:])
		    nlp[j]=logsumexp(fhat_test[j,:])-fhat_test[j,ytest[j]]
        end
        println("prop_missed=",1-sum(prediction.==ytest)/Ntest," mean_nlp=",mean(nlp)," theta norm=",norm(theta))
            
            mydata=(Symbol=>Any)[
	    :phi => phitrain,
	    :y => ytrain,
	    :n => n,
	    :C => C,
	    :N => Ntrain
            ]

            inits=[[:phi=>mydata[:phi], :y=> mydata[:y], :theta => reshape(theta,n,C)]]
            scheme=[Slice([:theta],ones(n*C))]
            setsamplers!(model,scheme)
            sim=mcmc(model,mydata,inits,10,burnin=9,thin=1,chains=1,verbose=false);
            samples=sim.value;
            theta=vec(samples[1,:,1]);
            
            
	    # M step - maximise joint log likelihood wrt hyperparams using no_cg_iter steps of cg/gd	
	    println("function_value_before:",f(theta,loghyperparams))
	    
        f2(loghyperparameters::Vector)=f(theta,loghyperparameters);
	    g2(loghyperparameters::Vector)=g(theta,loghyperparameters)[end-Lh+1:end];
	    function g2!(loghyperparameters::Vector,storage::Vector)
		grad=g2(loghyperparameters)
		for i=1:length(loghyperparameters)
	    	storage[i]=grad[i]
		end
	    end
	    l=Optim.optimize(f2,g2!,loghyperparams,method=:cg,show_trace = false, extended_trace = false, iterations=num_cg_iter)
	    new_loghyperparams=l.minimum
            #=
            new_loghyperparams=loghyperparams;
            for i=1:num_sg_iter
				loghypergrad=g(theta,new_loghyperparams)[end-Lh+1:end];
				gh=alpha*gh+(1-alpha)*(loghypergrad.^2)
				epsh=epsilonh./(sqrt(gh)+1e-5);
				println("epsh norm=",norm(epsh));
				new_loghyperparams-=epsh.*loghypergrad
				println("loghyperparam gradients:",loghypergrad);
			end
            =#
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

if 1==0
init_length_scale=ones(D)+0.2*randn(D);
init_sigma_RBF=1+0.2*randn();
hyperparams=[init_length_scale,init_sigma_RBF];
init_theta=randn(n*C);
GPNT_hyperparameters_ng(init_theta,[init_length_scale,init_sigma_RBF],neglogjointlkhd,gradneglogjointlkhd,Z,b)
end

if 1==1
init_length_scale=[2.1594,1.3297,1.3283,2.1715];
init_sigma_RBF=1.2459;
hyperparams=[init_length_scale,init_sigma_RBF];
init_theta=randn(n*C);
theta=init_theta;
#mytheta=Array(Float64,n*C);
#myepstheta=Array(Float64,n*C);
#mygtheta=Array(Float64,n*C);
#epsilont=1e-2;
nlp=Array(Float64,Ntest);
prediction=Array(Integer,Ntest);
gtheta=zeros(n*C);
epsilon=1e-2;
for i=1:3000
    gradtheta=gradneglogjointlkhd(theta,hyperparams,Z,b)[1:n*C];
    #gtheta=alpha*gtheta+(1-alpha)*(gradtheta.^2);
    #epstheta=epsilont./(sqrt(gtheta)+1e-5);
    #theta-=epstheta.*gradtheta/2+sqrt(epstheta).*randn(n*C);
    #mytheta=theta; myepstheta=epstheta; mygtheta=gtheta;
	mom=randn(n*C);
    theta_prop=theta-epsilon*gradtheta/2+sqrt(epsilon)*mom;	
    gradtheta_prop=gradneglogjointlkhd(theta_prop,hyperparams,Z,b)[1:n*C];
    mom_prop=mom-sqrt(epsilon)*(gradtheta+gradtheta_prop)/2;
    accept_prob=exp(neglogjointlkhd(theta,hyperparams,Z,b)-neglogjointlkhd(theta_prop,hyperparams,Z,b)+sum(mom.^2)/2-sum(mom_prop.^2)/2);
    if rand()<accept_prob
        theta=theta_prop
    end
    fhat_test=phitest'*reshape(theta,n,C);
    for j=1:Ntest
	prediction[j]=indmax(fhat_test[j,:])
	nlp[j]=logsumexp(fhat_test[j,:])-fhat_test[j,ytest[j]]
    end
    
    if i%100==0
        println("iter=",i," prop_missed=",1-sum(prediction.==ytest)/Ntest," mean_nlp=",mean(nlp)," theta norm=",norm(theta)," theta gradient norm=",norm(gradtheta)," accept_prob=",accept_prob);
    end
end

end

#=
    model=Model(
	
	theta=Stochastic(2,
		@modelexpr(n,C, Distribution[Normal(0,1) for i in 1:n, j in 1:C])
	),


	fhat=Logical(2,
		@modelexpr(theta,phi,phi'*theta),
		false
	),
	
	p=Logical(2,
		@modelexpr(fhat,N,C,
		[exp(fhat[i,c]-maximum(fhat[i,:])-log(sum(exp(fhat[i,:]-maximum(fhat[i,:]))))) for c=1:C,i=1:N]),
		false
	),
	fhat_test=Logical(2,
		@modelexpr(theta,phitest,phitest'*theta),
		false
	),
	prediction=Logical(1,
		@modelexpr(fhat_test,Ntest,[indmax(fhat_test[i,:]) for i=1:Ntest]),
		false
        ),
        nlp=Logical(1,
                @modelexpr(fhat_test,ytest,Ntest,[maximum(fhat_test[i,:])+log(sum(exp(fhat_test[i,:]-maximum(fhat_test[i,:]))))-fhat_test[i,ytest[i]] for i=1:Ntest]),
                false
        ),
	y=Stochastic(1,
		@modelexpr(p,N,
		Distribution[Categorical(p[:,i]) for i=1:N]
		),
	false
	),
        prop_missed=Logical(@modelexpr(prediction,ytest,Ntest,1-sum(prediction.==ytest)/Ntest)),
        mean_nlp=Logical(@modelexpr(nlp,mean(nlp)))

    )
            
            mydata=(Symbol=>Any)[
	    :phi => phitrain,
	    :phitest => phitest,
	    :y => ytrain,
	    :ytest => ytest,
	    :n => n,
	    :C => C,
	    :N => Ntrain,
	    :Ntest => Ntest
            ]

            inits=[[:phi=>mydata[:phi], :y=> mydata[:y], :theta => randn(n,C)]]
            scheme=[Slice([:theta],ones(n*C))]
            setsamplers!(model,scheme)
            sim=mcmc(model,mydata,inits,100,burnin=0,thin=1,chains=1,verbose=true);
            samples=sim.value;
=#

