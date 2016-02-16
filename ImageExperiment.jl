#include("GPkit.jl-master/src/GPkit.jl")

@everywhere using GPT_SGLD
@everywhere using DataFrames: readdlm
@everywhere using Mamba
@everywhere using Optim
#@everywhere using PyPlot
#@everywhere using Iterators: product

@everywhere data=readdlm("segment.dat",Float64);
@everywhere data=data[:,[1,2,6:20]] #3rd column is constant,4th&5th columns are not useful - remove
@everywhere N=size(data,1);
@everywhere D=16;
@everywhere C=7;
@everywhere Ntrain=1300;
@everywhere Ntest=N-Ntrain;
@everywhere seed=17;
@everywhere length_scale=[5.1212,1.4029,5.0614,5.8121,6.1082,4.7774,1.7421,1.6444,1.8365,1.7417,1.8233,2.8132,1.2788,1.8477,2.0961,1.1489]; @everywhere sigma_RBF=11.4468
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
@everywhere maxepoch=100;
@everywhere Q=200;
@everywhere m=50;
@everywhere r=10;
@everywhere n=5;
@everywhere numbatches=int(ceil(Ntrain/m))
@everywhere I=samplenz(r,D,Q,seed);
@everywhere scale=sqrt(n/(Q^(1/D)));
@everywhere phitrain=feature(Xtrain,n,length_scale,sigma_RBF,seed,scale);
@everywhere phitest=feature(Xtest,n,length_scale,sigma_RBF,seed,scale);
@everywhere epsw=1e-2; 
@everywhere epsU=1e-6;
#=
println("n=",n," m=",m," r=",r," Q=",Q," maxepoch=",maxepoch," epsw=",epsw," epsU=",epsU)
tic();w_store,U_store=GPT_SGDEclass(phitrain, ytrain, I, r, Q, m, epsw, epsU, burnin, maxepoch); toc();
prop_missed=Array(Float64,maxepoch);
nlp=Array(Float64,Ntest);
mean_nlp=Array(Float64,maxepoch);
fhat_test=Array(Float64,Ntest,C,maxepoch);
prediction=Array(Integer,Ntest);
for epoch=1:maxepoch
	for c=1:C
		fhat_test[:,c,epoch]=pred(w_store[:,c,numbatches*epoch],U_store[:,:,:,c,numbatches*epoch],I,phitest);
	end
	for i=1:Ntest
		prediction[i]=indmax(fhat_test[i,:,epoch])
		nlp[i]=logsumexp(fhat_test[i,:,epoch])-fhat_test[i,ytest[i],epoch]
	end
	prop_missed[epoch]=1-sum(prediction.==ytest)/Ntest
	mean_nlp[epoch]=mean(nlp)
end
subplot(2,1,1); plot(prop_missed,label=string("n=",n))
subplot(2,1,2); plot(mean_nlp,label=string("n=",n))

mean_fhat=squeeze(mean(fhat_test[:,:,51:100],3),3);
for i=1:Ntest
	prediction[i]=indmax(mean_fhat[i,:])
	nlp[i]=logsumexp(mean_fhat[i,:])-mean_fhat[i,ytest[i]]
end
final_prop_missed=1-sum(prediction.==ytest)/Ntest
final_mean_nlp=mean(nlp)
println("prop_missed with averaged pred=",final_prop_missed);
println("mean_nlp with averaged pred=",final_mean_nlp);
=#

#=
@everywhere t=Iterators.product(1:4,3:7)
@everywhere myt=Array(Any,20);
@everywhere it=1;
@everywhere for prod in t
	myt[it]=prod;
        it+=1;
        end
#myRMSE=SharedArray(Float64,70);
@sync @parallel for  Tuple in myt
    i,j=Tuple;
    epsw=float(string("1e-",i)); epsU=float(string("1e-",j));

	println("n=",n," m=",m," r=",r," Q=",Q," maxepoch=",maxepoch," epsw=",epsw," epsU=",epsU)
	tic();w_store,U_store=GPT_SGLDERMclass(phitrain, ytrain, I, r, Q, m, epsw, epsU, burnin, maxepoch); toc();
	prop_missed=Array(Float64,maxepoch);
	nlp=Array(Float64,Ntest);
	mean_nlp=Array(Float64,maxepoch);
	fhat_test=Array(Float64,Ntest,C,maxepoch);
	prediction=Array(Integer,Ntest);
	for epoch=1:maxepoch
		for c=1:C
			fhat_test[:,c,epoch]=pred(w_store[:,c,numbatches*epoch],U_store[:,:,:,c,numbatches*epoch],I,phitest);
		end
		for i=1:Ntest
			prediction[i]=indmax(fhat_test[i,:,epoch])
			nlp[i]=logsumexp(fhat_test[i,:,epoch])-fhat_test[i,ytest[i],epoch]
		end
		prop_missed[epoch]=1-sum(prediction.==ytest)/Ntest
		mean_nlp[epoch]=mean(nlp)
	end
	println("last epoch: prop_missed=",prop_missed[maxepoch]," mean_nlp=",mean_nlp[maxepoch])
	#figure()
	#plot(trainRMSE)
	#subplot(2,1,1); plot(prop_missed,label=string("n=",n))
	#subplot(2,1,2); plot(mean_nlp,label=string("n=",n))
	
end


mean_fhat=squeeze(mean(fhat_test[:,:,51:100],3),3);
for i=1:Ntest
	prediction[i]=indmax(mean_fhat[i,:])
	nlp[i]=logsumexp(mean_fhat[i,:])-mean_fhat[i,ytest[i]]
end
final_prop_missed=1-sum(prediction.==ytest)/Ntest
final_mean_nlp=mean(nlp)
println("prop_missed with averaged pred=",final_prop_missed);
println("mean_nlp with averaged pred=",final_mean_nlp);
=#



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
neglogjointlkhd::Function,gradneglogjointlkhd::Function;epsilont::Real=1e-2,epsilonh::Real=1e-5,num_hmc_iter::Integer=10,num_cg_iter::Integer=10,alpha::Real=0.9)
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
	    
            length_scale=exp(loghyperparams[1:Lh-1]);sigma_RBF=exp(loghyperparams[Lh]);
	    phitrain=featureNotensor(Xtrain,n,length_scale,sigma_RBF,seed);
            phitest=featureNotensor(Xtest,n,length_scale,sigma_RBF,seed);
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
            scheme=[NUTS([:theta])]
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
init_length_scale=ones(D)+0.2*randn(D);
init_sigma_RBF=1+0.2*randn(1)[1];
hyperparams=[init_length_scale,init_sigma_RBF];
init_theta=randn(n*C);
testng(init_theta,[init_length_scale,init_sigma_RBF],neglogjointlkhd,gradneglogjointlkhd)




