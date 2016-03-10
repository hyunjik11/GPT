@everywhere using DataFrames
@everywhere using GPT_SGLD
@everywhere using Distributions
@everywhere using HDF5
@everywhere using Iterators

### data processing
@everywhere function getdummy{R}(df::DataFrame, cname::Symbol, ::Type{R})
    darr = df[cname]
    vals = sort(levels(darr))#[2:end]
    namedict = Dict(vals, 1:length(vals))   
    arr = zeros(R, length(darr), length(namedict))
    for i=1:length(darr)
        if haskey(namedict, darr[i])
            arr[i, namedict[darr[i]]] = 1
        end        
    end
    newdf = convert(DataFrame, arr)
    names!(newdf, [symbol("$(cname)_$k") for k in vals])
    return newdf
end

@everywhere function convertdummy{R}(df::DataFrame, cnames::Array{Symbol}, ::Type{R})
    # consider every variable from cnames as categorical
    # and convert them into set of dummy variables,
    # return new dataframe
    newdf = DataFrame()
    for cname in names(df)
        if !in(cname, cnames)
            newdf[cname] = df[cname]
        else
            dummydf = getdummy(df, cname, R)
            for dummyname in names(dummydf)
                newdf[dummyname] = dummydf[dummyname]
            end
        end
    end
    return newdf
end

@everywhere convertdummy(df::DataFrame, cnames::Array{Symbol}) = convertdummy(df, cnames, Int32)

@everywhere function bin_age(age::Array)
	q=quantile(age,[0.2,0.4,0.6,0.8,1.0])
	indmin(q.<UserData[30,2])
        map(x->indmin(q.<x),age)
end

##data clearing
@everywhere UserData = readdlm("ml-100k/u.user", '|');
@everywhere MovieData = readdlm("ml-100k/u.item",'|');
@everywhere Rating = readdlm("ml-100k/u.data",Float64);

@everywhere Ntrain = 80000;
@everywhere Ntest = 20000;
@everywhere UserData[:,2] = bin_age(UserData[:,2])
@everywhere UserData = convertdummy(convert(DataFrame,UserData),[:x2,:x3,:x4])[:,1:end-1];
@everywhere MovieData = MovieData[:,[1,6:end]];
@everywhere UserData = convert(Array{Float64,2},UserData)[:,2:end];
@everywhere MovieData = convert(Array{Float64,2},MovieData)[:,3:end]; 
@everywhere n1,D1=size(UserData); 
@everywhere n2,D2=size(MovieData); 
@everywhere ytrain = Rating[1:Ntrain,3];
@everywhere ytest = Rating[Ntrain+1:Ntrain+Ntest,3];
@everywhere ytrainMean=mean(ytrain);
@everywhere ytrainStd=std(ytrain);
@everywhere ytrain=(ytrain-ytrainMean)/ytrainStd;
@everywhere ytest = (ytest-ytrainMean)/ytrainStd;
@everywhere Ratingtrain=hcat(Rating[1:Ntrain,1:2],ytrain);
@everywhere Ratingtest=hcat(Rating[Ntrain+1:Ntrain+Ntest,1:2],ytest);
@everywhere n = 30; 
@everywhere M = 5;
@everywhere burnin=0;
@everywhere numiter=30;
@everywhere r = 16
@everywhere Q=r;   
@everywhere D = 2;
@everywhere signal_var = 0.5;
@everywhere param_seed=17;
@everywhere I=repmat(1:r,1,2);
@everywhere m = 100;
@everywhere maxepoch = 100;
@everywhere epsw=1e-2#2e-2
@everywhere epsU=5e-7#4e-10
@everywhere sigma_w=sqrt(n1*n2)/r
@everywhere sigma_u=0.1;

@everywhere numbatches=int(ceil(maximum(Ntrain)/m));
@everywhere a=1;b1=1;b2=2;
@everywhere UserHashmap=Array(Int64,M,n1); 
@everywhere MovieHashmap=Array(Int64,M,n2);
@everywhere for i=1:n1
	UserHashmap[:,i]=sample(1:n,M,replace=false)
end
@everywhere for i=1:n2
	MovieHashmap[:,i]=sample(1:n,M,replace=false)
end
@everywhere UserBHashmap=2*rand(Bernoulli(),M,n1)-1
@everywhere MovieBHashmap=2*rand(Bernoulli(),M,n2)-1
@everywhere a=1;
@everywhere b1=1;
@everywhere b2=1;

# function to cutoff predictions at 1 and 5
@everywhere function cutoff!(pred::Vector)
	idxlow=(pred.<1); pred[idxlow]=1;
	idxhigh=(pred.>5); pred[idxhigh]=5;
end

# function for tensor model for CF with no side information, using full W
function GPT_test(Rating::Array,UserData::Array,MovieData::Array,Ratingtest::Array, signal_var::Real, sigma_w::Real, r::Integer, m::Integer, epsw::Real, epsU::Real,burnin::Integer, maxepoch::Integer, param_seed::Integer; langevin::Bool=false, stiefel::Bool=true)
	N=size(Rating,1);Ntest=size(Ratingtest,1);
	n1,D1=size(UserData); 
	n2,D2=size(MovieData);
	numbatches=int(ceil(N/m));
	
	# initialise w,U,V
	srand(param_seed);
	w_store=Array(Float64,r,r,maxepoch)
	U_store=Array(Float64,n1,r,maxepoch)
	V_store=Array(Float64,n2,r,maxepoch)
	w=sigma_w*randn(r,r);
	if stiefel
		Z1=randn(r,n1);	Z2=randn(r,n2)
		U=transpose(\(sqrtm(Z1*Z1'),Z1))
		V=transpose(\(sqrtm(Z2*Z2'),Z2))
    else U=randn(n1,r)/sqrt(n1);V=randn(n2,r)/sqrt(n2)
    end
	
	testpred=zeros(Ntest)
	counter=0;
	for epoch=1:(burnin+maxepoch)
		#randomly permute training data and divide into mini_batches of size m
        perm=randperm(N)
		shuffledRatings=Rating[perm,:]
		
		#run SGLD on w and U
		for batch=1:numbatches
            # random samples for the stochastic gradient
            idx=(m*(batch-1)+1):min(m*batch,N);
			batch_size=length(idx);
			batch_ratings=shuffledRatings[idx,:];
			
			# compute gradients
			gradw=zeros(r,r);
			gradU=zeros(n1,r);
			gradV=zeros(n2,r);
			for ii=1:batch_size
				user=batch_ratings[ii,1]; movie=batch_ratings[ii,2]; rating=batch_ratings[ii,3];
				pred=sum((U[user,:]*w).*V[movie,:])
				gradw+=(rating-pred)*U[user,:]'*V[movie,:]/(batch_size*signal_var)
				gradU[user,:]+=(rating-pred)*V[movie,:]*w'/signal_var
				gradV[movie,:]+=(rating-pred)*U[user,:]*w/signal_var
			end
			gradw*=N/batch_size; gradw-=w/sigma_w^2;
			gradU*=N/batch_size;
			gradV*=N/batch_size;

			# update w
			if langevin
				w+=epsw*gradw/2+sqrt(epsw)*randn(r,r)
			else w+=epsw*gradw/2
			end
			
			# update U,V
			if langevin
				if stiefel
				        momU=proj(U,sqrt(epsU)*gradU/2+randn(n1,r)); momV=proj(V,sqrt(epsU)*gradV/2+randn(n2,r));
				        U=geod(U,momU,sqrt(epsU)); V=geod(V,momV,sqrt(epsU));
				        if U==zeros(n1,r) || V==zeros(n2,r)#if NaN appears while evaluating G
				            return zeros(r,r,maxepoch),zeros(n1,r,maxepoch),zeros(n2,r,maxepoch)
				        end
				else U+=epsU*(gradU-n1*U)/2+sqrt(epsU)*randn(n1,r); V+=epsU*(gradV-n2*V)/2+sqrt(epsU)*randn(n2,r);
				end
			else
				if stiefel
				        momU=proj(U,sqrt(epsU)*gradU/2); momV=proj(V,sqrt(epsU)*gradV/2);
				        U=geod(U,momU,sqrt(epsU)); V=geod(V,momV,sqrt(epsU));
				        if U==zeros(n1,r) || V==zeros(n2,r)#if NaN appears while evaluating G
				            return zeros(r,r,maxepoch),zeros(n1,r,maxepoch),zeros(n2,r,maxepoch)
				        end
				else U+=epsU*(gradU-n1*U)/2; V+=epsU*(gradV-n2*V)/2;
				end
			end
		end
		
		if epoch>burnin
			w_store[:,:,epoch-burnin]=w
			U_store[:,:,epoch-burnin]=U
			V_store[:,:,epoch-burnin]=V
		
			trainpred=Array(Float64,N)
			for i=1:N
				user=Rating[i,1]; movie=Rating[i,2]; 
				trainpred[i]=sum((U[user,:]*w).*V[movie,:])
			end
			final_trainpred=trainpred*ytrainStd+ytrainMean;
			cutoff!(final_trainpred);
			trainRMSE=sqrt(sum((ytrainStd*Rating[:,3]+ytrainMean-final_trainpred).^2)/N)
			
			#counter=0;
			for i=1:Ntest
				user=Ratingtest[i,1]; movie=Ratingtest[i,2];
				testpred[i]=(testpred[i]*counter+sum((U[user,:]*w).*V[movie,:]))/(counter+1)
			end
			final_testpred=testpred*ytrainStd+ytrainMean;
			cutoff!(final_testpred);
			testRMSE=sqrt(sum((ytrainStd*Ratingtest[:,3]+ytrainMean-final_testpred).^2)/Ntest)
			counter+=1
			println("epoch=$epoch, trainRMSE=$trainRMSE, testRMSE=$testRMSE")
		end
	end
	return w_store,U_store,V_store
end

# function for tensor model for CF with no side information, using w=I.
function GPT_pmf(Rating::Array,UserData::Array,MovieData::Array,Ratingtest::Array, signal_var::Real, sigma_u::Real, r::Integer, m::Integer, epsU::Real,burnin::Integer, maxepoch::Integer, param_seed::Integer)
	N=size(Rating,1);Ntest=size(Ratingtest,1);
	n1,D1=size(UserData); 
	n2,D2=size(MovieData);
	numbatches=int(ceil(N/m));
	
	# initialise U,V
	srand(param_seed);
	U_store=Array(Float64,n1,r,maxepoch)
	V_store=Array(Float64,n2,r,maxepoch)
	U=sigma_u*randn(n1,r);V=sigma_u*randn(n2,r);
    
	testpred=zeros(Ntest)
	counter=0;
	for epoch=1:(burnin+maxepoch)
		#randomly permute training data and divide into mini_batches of size m
        perm=randperm(N)
		shuffledRatings=Rating[perm,:]
		
		#run SGLD on w and U
		for batch=1:numbatches
            # random samples for the stochastic gradient
            idx=(m*(batch-1)+1):min(m*batch,N);
			batch_size=length(idx);
			batch_ratings=shuffledRatings[idx,:];
			
			# compute gradients
			gradU=zeros(n1,r);
			gradV=zeros(n2,r);
			for ii=1:batch_size
				user=batch_ratings[ii,1]; movie=batch_ratings[ii,2]; rating=batch_ratings[ii,3];
				pred=sum(U[user,:].*V[movie,:])
				gradU[user,:]+=(rating-pred)*V[movie,:]/signal_var
				gradV[movie,:]+=(rating-pred)*U[user,:]/signal_var
			end
			gradU*=N/batch_size;
			gradV*=N/batch_size;

			# update U,V
			U+=epsU*(gradU-U/sigma_u^2)/2; V+=epsU*(gradV-V/sigma_u^2)/2;
		end
		
		if epoch>burnin
			U_store[:,:,epoch-burnin]=U
			V_store[:,:,epoch-burnin]=V
		
			trainpred=Array(Float64,N)
			for i=1:N
				user=Rating[i,1]; movie=Rating[i,2]; 
				trainpred[i]=sum(U[user,:].*V[movie,:])
			end
			final_trainpred=trainpred*ytrainStd+ytrainMean;
			cutoff!(final_trainpred);
			trainRMSE=sqrt(sum((ytrainStd*Rating[:,3]+ytrainMean-final_trainpred).^2)/N)
			
			#counter=0;
			for i=1:Ntest
				user=Ratingtest[i,1]; movie=Ratingtest[i,2];
				testpred[i]=(testpred[i]*counter+sum(U[user,:].*V[movie,:]))/(counter+1)
			end
			final_testpred=testpred*ytrainStd+ytrainMean;
			cutoff!(final_testpred);
			testRMSE=sqrt(sum((ytrainStd*Ratingtest[:,3]+ytrainMean-final_testpred).^2)/Ntest)
			counter+=1
			println("epoch=$epoch, trainRMSE=$trainRMSE, testRMSE=$testRMSE")
		end
	end
	return U_store,V_store
end

# function for tensor model for CF with no side information, using full w - initialised by draw from distrib induced by priors on each row of U,V
# use instead U~N(0,1/n1) and V~N(0,1/n2), and multiply w by sqrt(n1*n2) st can use same epsU for both Stiefel=true and Stiefel=false
@everywhere function GPT_fullw(Rating::Array,UserData::Array,MovieData::Array,Ratingtest::Array, signal_var::Real, w_init::Array, m::Integer, epsw::Real, epsU::Real,burnin::Integer, maxepoch::Integer, param_seed::Integer; langevin::Bool=false, stiefel::Bool=false)
	N=size(Rating,1);Ntest=size(Ratingtest,1);
	n1,D1=size(UserData); 
	n2,D2=size(MovieData);
	numbatches=int(ceil(N/m));
	r=size(w_init,1);
	
	# initialise w,U,V
	srand(param_seed);
	w_store=Array(Float64,r,r,maxepoch)
	U_store=Array(Float64,n1,r,maxepoch)
	V_store=Array(Float64,n2,r,maxepoch)
	w=w_init*sqrt(n1*n2);
	sigma_w=sqrt(sum(w.^2))/r;
	if stiefel
		Z1=randn(r,n1);	Z2=randn(r,n2)
		U=transpose(\(sqrtm(Z1*Z1'),Z1))
		V=transpose(\(sqrtm(Z2*Z2'),Z2))
    else U=randn(n1,r)/sqrt(n1);V=randn(n2,r)/sqrt(n2);
		U[:,r]=1/sqrt(n1); V[:,r]=1/sqrt(n2);
    end
	
	trainRMSEvec=Array(Float64,maxepoch)
	testRMSEvec=Array(Float64,maxepoch)
	testpred=zeros(Ntest)
	counter=0;
	for epoch=1:(burnin+maxepoch)
		#randomly permute training data and divide into mini_batches of size m
        perm=randperm(N)
		shuffledRatings=Rating[perm,:]
		
		#run SGLD on w and U
		for batch=1:numbatches
            # random samples for the stochastic gradient
            idx=(m*(batch-1)+1):min(m*batch,N);
			batch_size=length(idx);
			batch_ratings=shuffledRatings[idx,:];
			
			# compute gradients
			gradw=zeros(r,r);
			gradU=zeros(n1,r);
			gradV=zeros(n2,r);
			for ii=1:batch_size
				user=batch_ratings[ii,1]; movie=batch_ratings[ii,2]; rating=batch_ratings[ii,3];
				pred=sum((U[user,:]*w).*V[movie,:])
				gradw+=(rating-pred)*U[user,:]'*V[movie,:]/(batch_size*signal_var)
				gradU[user,:]+=(rating-pred)*V[movie,:]*w'/signal_var
				gradV[movie,:]+=(rating-pred)*U[user,:]*w/signal_var
			end
			gradw*=N/batch_size; gradw-=w/sigma_w^2;
			gradU*=N/batch_size;
			gradV*=N/batch_size;

			# update w
			if langevin
				w+=epsw*gradw/2+sqrt(epsw)*randn(r,r)
			else w+=epsw*gradw/2
			end
			
			# update U,V
			if langevin
				if stiefel
				        momU=proj(U,sqrt(epsU)*gradU/2+randn(n1,r)); momV=proj(V,sqrt(epsU)*gradV/2+randn(n2,r));
				        U=geod(U,momU,sqrt(epsU)); V=geod(V,momV,sqrt(epsU));
				        if U==zeros(n1,r) || V==zeros(n2,r)#if NaN appears while evaluating G
				            return zeros(r,r,maxepoch),zeros(n1,r,maxepoch),zeros(n2,r,maxepoch)
				        end
				else U+=epsU*(gradU-n1*U)/2+sqrt(epsU)*randn(n1,r); V+=epsU*(gradV-n2*V)/2+sqrt(epsU)*randn(n2,r);
				end
			else
				if stiefel
				        momU=proj(U,sqrt(epsU)*gradU/2); momV=proj(V,sqrt(epsU)*gradV/2);
				        U=geod(U,momU,sqrt(epsU)); V=geod(V,momV,sqrt(epsU));
				        if U==zeros(n1,r) || V==zeros(n2,r)#if NaN appears while evaluating G
				            return zeros(r,r,maxepoch),zeros(n1,r,maxepoch),zeros(n2,r,maxepoch)
				        end
				else U+=epsU*(gradU-n1*U)/2; V+=epsU*(gradV-n2*V)/2;
				end
			end
		end
		
		if epoch>burnin
			w_store[:,:,epoch-burnin]=w
			U_store[:,:,epoch-burnin]=U
			V_store[:,:,epoch-burnin]=V
		
			trainpred=Array(Float64,N)
			for i=1:N
				user=Rating[i,1]; movie=Rating[i,2]; 
				trainpred[i]=sum((U[user,:]*w).*V[movie,:])
			end
			final_trainpred=trainpred*ytrainStd+ytrainMean;
			cutoff!(final_trainpred);
			trainRMSE=sqrt(sum((ytrainStd*Rating[:,3]+ytrainMean-final_trainpred).^2)/N)
			trainRMSEvec[epoch-burnin]=trainRMSE
			counter=0;
			for i=1:Ntest
				user=Ratingtest[i,1]; movie=Ratingtest[i,2];
				testpred[i]=(testpred[i]*counter+sum((U[user,:]*w).*V[movie,:]))/(counter+1)
			end
			final_testpred=testpred*ytrainStd+ytrainMean;
			cutoff!(final_testpred);
			testRMSE=sqrt(sum((ytrainStd*Ratingtest[:,3]+ytrainMean-final_testpred).^2)/Ntest)
			testRMSEvec[epoch-burnin]=testRMSE
			counter+=1
			println("epoch=$epoch, trainRMSE=$trainRMSE, testRMSE=$testRMSE")
		end
	end
	return w_store,U_store,V_store,trainRMSEvec,testRMSEvec
end

# function for tensor model for CF with no side information, using full w - initialised by draw from distrib induced by priors on each row of U,V, then using Gibbs sampling for U,V,w
@everywhere function GPT_fullw_gibbs(Rating::Array,UserData::Array,MovieData::Array,Ratingtest::Array, signal_var::Real, w_init::Array, burnin::Integer, maxepoch::Integer, n_samples::Integer, param_seed::Integer)
	N=size(Rating,1);Ntest=size(Ratingtest,1);
	n1,D1=size(UserData); 
	n2,D2=size(MovieData);
	r=size(w_init,1);
	
	# initialise w,U,V
	srand(param_seed);
	w_store=Array(Float64,r,r,maxepoch)
	U_store=Array(Float64,n1,r,maxepoch)
	V_store=Array(Float64,n2,r,maxepoch)
	w=w_init;
	sigma_w=sqrt(sum(w.^2))/r;
	sigma_U=1;
	U=randn(n1,r);V=randn(n2,r);
	U[:,r]=1; V[:,r]=1;
	U*=sigma_U; V*=sigma_U;

	trainRMSEvec=Array(Float64,maxepoch)
	testRMSEvec=Array(Float64,maxepoch)
	testpred=zeros(Ntest)
	counter=0;
	Kron=Array(Float64,N,r^2);
	for epoch=1:(burnin+maxepoch)
		for gibbs=1:n_samples		
			# update U
			for i=1:n1
				idx=(Rating[:,1].==i) #BitArray size N
				if 	sum(idx)>0		
					Ni=Rating[idx,2] #vector of movies rated by user i
					yi=Rating[idx,3] #vector of these ratings
					VNiw=V[Ni,:]*w';
					invSigmai=VNiw'*VNiw/signal_var+eye(r)/(sigma_U^2);
					mui=\(invSigmai,VNiw'*yi)/signal_var;
					U[i,:]=\(chol(invSigmai,:U),randn(r))+mui
				#else println("user $i has not rated any movies in training set")
				end
			end
		
			# update V
			for j=1:n2
				idx=(Rating[:,2].==j) #BitArray size N
				if 	sum(idx)>0		
					Nj=Rating[idx,1] #vector of users who rated movie j
					yj=Rating[idx,3] #vector of these ratings
					UNjw=U[Nj,:]*w;
					invSigmaj=UNjw'*UNjw/signal_var+eye(r)/(sigma_U^2);
					muj=\(invSigmaj,UNjw'*yj)/signal_var;
					V[j,:]=\(chol(invSigmaj,:U),randn(r))+muj
				#else println("movie $j has not been rated in training set")
				end
			end
		
			# update w
			for ii=1:N
				user=Rating[ii,1];movie=Rating[ii,2];
				Kron[ii,:]=kron(V[movie,:],U[user,:]);
			end
			invSigma=Kron'*Kron/signal_var+eye(r^2)/(sigma_w^2);
			mu=\(invSigma,Kron'*Rating[:,3])/signal_var;
			w[:]=\(chol(invSigma,:U),randn(r^2))+mu;
		end
		if epoch>burnin
			w_store[:,:,epoch-burnin]=w
			U_store[:,:,epoch-burnin]=U
			V_store[:,:,epoch-burnin]=V
		
			trainpred=Array(Float64,N)
			for i=1:N
				user=Rating[i,1]; movie=Rating[i,2]; 
				trainpred[i]=sum((U[user,:]*w).*V[movie,:])
			end
			final_trainpred=trainpred*ytrainStd+ytrainMean;
			cutoff!(final_trainpred);
			trainRMSE=sqrt(sum((ytrainStd*Rating[:,3]+ytrainMean-final_trainpred).^2)/N)
			trainRMSEvec[epoch-burnin]=trainRMSE
			#counter=0;
			for i=1:Ntest
				user=Ratingtest[i,1]; movie=Ratingtest[i,2];
				testpred[i]=(testpred[i]*counter+sum((U[user,:]*w).*V[movie,:]))/(counter+1)
			end
			final_testpred=testpred*ytrainStd+ytrainMean;
			cutoff!(final_testpred);
			testRMSE=sqrt(sum((ytrainStd*Ratingtest[:,3]+ytrainMean-final_testpred).^2)/Ntest)
			testRMSEvec[epoch-burnin]=testRMSE
			counter+=1
			println("epoch=$epoch, trainRMSE=$trainRMSE, testRMSE=$testRMSE")
		end
	end
	return w_store,U_store,V_store,trainRMSEvec,testRMSEvec
end



@everywhere file="ml100k_UVhyperparams.h5";
@everywhere mu_u=h5read(file,"mu_u");
@everywhere mu_m=h5read(file,"mu_m");
@everywhere var_u=h5read(file,"var_u");
@everywhere var_m=h5read(file,"var_m");
@everywhere tempr=length(mu_u);
@everywhere r=tempr+1;
@everywhere Lu=chol(var_u); 
@everywhere Lm=chol(var_m); # so mu_u+Lu'*randn(tempr) gives N(mu_u,var_u)
@everywhere w_init=Array(Float64,tempr+1,tempr+1);
@everywhere w_init[1:tempr,1:tempr]=Lu'*Lm;
@everywhere w_init[r,1:tempr]=mu_u'*Lu';
@everywhere w_init[1:tempr,r]=Lu*mu_m;
@everywhere w_init[r,r]=sum(mu_u.*mu_m);

#epsw=3e+1;
#epsU=1e-10;
#=
@everywhere t=Iterators.product(-3:1:0,0:1:4,-1:1:2,-12:1:-7)
@everywhere myt=Array(Any,480);
@everywhere it=1;
@everywhere for prod in t
	myt[it]=prod;
        it+=1;
end


@parallel for  Tuple in myt 
    i,j,k,l=Tuple;
    signal_var=float(string("1e",i));
	sigma_w=float(string("1e",j));
	epsw=float(string("1e",k));
	epsU=float(string("1e",l));
	w_store,U_store,V_store,trainRMSEvec,testRMSEvec=GPT_fullw(Ratingtrain,UserData,MovieData,Ratingtest,signal_var,sigma_w,w_init, m, epsw, epsU, burnin, maxepoch, param_seed,stiefel=false,langevin=false);
	mintrain=minimum(trainRMSEvec); mintest=minimum(testRMSEvec);
	println("mintest=$mintest,mintrain=$mintrain,epsw=$epsw,epsU=$epsU,signal_var=$signal_var,sigma_w=$sigma_w")
end
=#
signal_var=0.1; epsw=10; epsU=1e-11; burnin=10; maxepoch=500; n_samples=8;
#w_store,U_store,V_store,trainRMSEvec,testRMSEvec=GPT_fullw_gibbs(Ratingtrain,UserData,MovieData,Ratingtest,signal_var,w_init, burnin, maxepoch, n_samples, param_seed);
#println("")
#w_store,U_store,V_store,trainRMSEvec,testRMSEvec=GPT_fullw(Ratingtrain,UserData,MovieData,Ratingtest,signal_var,w_init, m, epsw, epsU, burnin, maxepoch, param_seed,stiefel=true,langevin=false);
#GPT_pmf(Ratingtrain,UserData,MovieData,Ratingtest,signal_var, sigma_u,r, m, epsU,burnin, maxepoch, param_seed)
#w_store,U_store,V_store=GPT_test(Ratingtrain,UserData,MovieData,Ratingtest,signal_var, sigma_w, r, m, epsw, epsU, burnin, maxepoch, param_seed,stiefel=false,langevin=false);

#randfeature(hyperparams::Vector)=CFfeatureNotensor(Ratingtrain,UserData,MovieData,UserHashmap,MovieHashmap, UserBHashmap,MovieBHashmap,n,hyperparams[1],hyperparams[2],hyperparams[3]);
#gradfeature(hyperparams::Vector)=CFgradfeatureNotensor(Ratingtrain,UserData,MovieData,UserHashmap,MovieHashmap, UserBHashmap,MovieBHashmap,n,hyperparams[1],hyperparams[2],hyperparams[3]);
#phiUser,phiMovie=CFfeature(UserData,MovieData,UserHashmap,MovieHashmap, UserBHashmap,MovieBHashmap,n,a,b1,b2);
#phiUser=phiUser[1:n,:];
#norm[l]=sum((phiUser'*phiUser-eye(Nu)).^2);
#println("M=$M, norm=$norm")
#println("mean(norm)=",mean(norm),"std(norm)=",std(norm));
#@time Z2=CFfeatureNotensor(Ratingtrain,UserData,MovieData,UserHashmap,MovieHashmap, UserBHashmap,MovieBHashmap,n,a,b1,b2);
#@time Z3=CFgradfeatureNotensor(Ratingtrain,UserData,MovieData,UserHashmap,MovieHashmap, UserBHashmap,MovieBHashmap,n,a,b1,b2);
#nlogmarginal(hyperparams::Vector)=GPNT_nlogmarginal(ytrain,(n+Du)*(n+Dm),hyperparams,randfeature)
#gradnlogmarginal(hyperparams::Vector)=GPNT_gradnlogmarginal(ytrain,(n+Du)*(n+Dm),hyperparams,randfeature,gradfeature)
#GPNT_hyperparameters(nlogmarginal,gradnlogmarginal,[0.1,0.1,1,0.1],0.001*ones(4),alg=:LD_LBFGS)

###results:
#mintest=1.1180241428095605,mintrain=1.1265601067684456,epsw=0.1,epsU=1.0e-12,signal_var=0.01,sigma_w=1.0
#mintest=1.1185931955306565,mintrain=1.1273102864615163,epsw=0.1,epsU=1.0e-12,signal_var=0.1,sigma_w=1.0
#mintest=1.1163529807513988,mintrain=1.1234949051567995,epsw=1.0,epsU=1.0e-12,signal_var=0.1,sigma_w=10.0
#mintest=1.1182766572790301,mintrain=1.127094734649239,epsw=100.0,epsU=1.0e-11,signal_var=1.0,sigma_w=10.0
#mintest=1.1186668832590634,mintrain=1.1274006713807245,epsw=0.1,epsU=1.0e-12,signal_var=1.0,sigma_w=1.0
#mintest=1.1181824511445937,mintrain=1.1266582145069786,epsw=1.0,epsU=1.0e-12,signal_var=1.0,sigma_w=10.0
#mintest=1.115579938804368,mintrain=1.1208611140759108,epsw=10.0,epsU=1.0e-12,signal_var=1.0,sigma_w=100.0<--
#mintest=1.115714163018838,mintrain=1.12009099577864,epsw=0.1,epsU=1.0e-12,signal_var=0.01,sigma_w=10.0
#mintest=1.1166891592015502,mintrain=1.1239341901834632,epsw=0.1,epsU=1.0e-12,signal_var=0.1,sigma_w=10.0
#mintest=1.1181720173073841,mintrain=1.126898812445316,epsw=1.0,epsU=1.0e-11,signal_var=0.01,sigma_w=1.0
#mintest=1.1182496481755313,mintrain=1.1267406810330296,epsw=0.1,epsU=1.0e-12,signal_var=1.0,sigma_w=10.0
#mintest=1.1163438491980309,mintrain=1.120655700609109,epsw=1.0,epsU=1.0e-12,signal_var=0.1,sigma_w=100.0
#mintest=1.1185310217932019,mintrain=1.1272443419496314,epsw=1.0,epsU=1.0e-11,signal_var=0.1,sigma_w=1.0
#mintest=1.1179935066243638,mintrain=1.1265516703558125,epsw=10.0,epsU=1.0e-11,signal_var=1.0,sigma_w=10.0
#mintest=1.1186598715556504,mintrain=1.1273924236378874,epsw=1.0,epsU=1.0e-11,signal_var=1.0,sigma_w=1.0
#mintest=1.1161233483210973,mintrain=1.1206655504332443,epsw=1.0,epsU=1.0e-12,signal_var=1.0,sigma_w=100.0
#mintest=1.1161305046384056,mintrain=1.120675471088067,epsw=10.0,epsU=1.0e-12,signal_var=1.0,sigma_w=1000.0
#mintest=1.1185976559137667,mintrain=1.1273129520617868,epsw=0.1,epsU=1.0e-11,signal_var=0.1,sigma_w=1.0
#mintest=1.118666954056236,mintrain=1.127400714765775,epsw=0.1,epsU=1.0e-11,signal_var=1.0,sigma_w=1.0
#mintest=1.1157356502772295,mintrain=1.1208916377374285,epsw=10.0,epsU=1.0e-11,signal_var=1.0,sigma_w=100.0<--
#mintest=1.1164622334766858,mintrain=1.1206497729013607,epsw=1.0,epsU=1.0e-12,signal_var=0.1,sigma_w=1000.0
#mintest=1.1164417371086883,mintrain=1.1237080723384518,epsw=1.0,epsU=1.0e-11,signal_var=0.1,sigma_w=10.0
#mintest=1.11825489280877,mintrain=1.1270553647241817,epsw=100.0,epsU=1.0e-10,signal_var=1.0,sigma_w=10.0
#mintest=1.1171144849960297,mintrain=1.1205691699235711,epsw=1.0,epsU=1.0e-12,signal_var=1.0,sigma_w=1000.0
#mintest=1.1182009976582845,mintrain=1.1266836713776391,epsw=1.0,epsU=1.0e-11,signal_var=1.0,sigma_w=10.0
#mintest=1.1161399131650926,mintrain=1.120675667751874,epsw=10.0,epsU=1.0e-12,signal_var=1.0,sigma_w=10000.0
#mintest=1.1173100775608131,mintrain=1.1254393186755054,epsw=0.1,epsU=1.0e-11,signal_var=0.1,sigma_w=10.0
#mintest=1.1185255902628253,mintrain=1.1272499527785735,epsw=1.0,epsU=1.0e-10,signal_var=0.1,sigma_w=1.0
#mintest=1.118332706182416,mintrain=1.1268765435862917,epsw=0.1,epsU=1.0e-11,signal_var=1.0,sigma_w=10.0
#mintest=1.118659905018143,mintrain=1.1273927771994816,epsw=1.0,epsU=1.0e-10,signal_var=1.0,sigma_w=1.0
#mintest=1.1164508972516172,mintrain=1.1206485698424118,epsw=10.0,epsU=1.0e-11,signal_var=1.0,sigma_w=1000.0
#mintest=1.1164634959680013,mintrain=1.120649746079317,epsw=1.0,epsU=1.0e-12,signal_var=0.1,sigma_w=10000.0
#mintest=1.1180276293778721,mintrain=1.1265691821094421,epsw=10.0,epsU=1.0e-10,signal_var=1.0,sigma_w=10.0
#mintest=1.1180131924807528,mintrain=1.1171214485921737,epsw=1.0,epsU=1.0e-11,signal_var=0.1,sigma_w=100.0
#mintest=1.1171430674197076,mintrain=1.1205803745330165,epsw=1.0,epsU=1.0e-12,signal_var=1.0,sigma_w=10000.0
#mintest=1.1195684282539373,mintrain=1.1223744403565885,epsw=1.0,epsU=1.0e-11,signal_var=1.0,sigma_w=100.0
#mintest=1.1182903097670267,mintrain=1.126822456885704,epsw=1.0,epsU=1.0e-10,signal_var=1.0,sigma_w=100
#mintest=1.1157090045891302,mintrain=1.1200925145515683,epsw=10.0,epsU=1.0e-10,signal_var=1.0,sigma_w=100.0<--
#mintest=1.1186674237187906,mintrain=1.1274011036788214,epsw=0.1,epsU=1.0e-10,signal_var=1.0,sigma_w=1.0
#mintest=1.1164634643316542,mintrain=1.120648270451184,epsw=10.0,epsU=1.0e-11,signal_var=1.0,sigma_w=10000.0<---
#mintest=1.1182585838164658,mintrain=1.1270658108664464,epsw=1.0,epsU=1.0e-12,signal_var=0.01,sigma_w=1.0
#mintest=1.1181433377596748,mintrain=1.1268340487354513,epsw=100.0,epsU=1.0e-9,signal_var=1.0,sigma_w=10.0
#mintest=1.1185314900036762,mintrain=1.127241390384816,epsw=1.0,epsU=1.0e-12,signal_var=0.1,sigma_w=1.0
#mintest=1.118278970496825,mintrain=1.1270989062984484,epsw=100.0,epsU=1.0e-12,signal_var=1.0,sigma_w=10.0
#mintest=1.1186598679560729,mintrain=1.1273923876547025,epsw=1.0,epsU=1.0e-12,signal_var=1.0,sigma_w=1.0
#mintest=1.1193456983266452,mintrain=1.1154252790038386,epsw=1.0,epsU=1.0e-10,signal_var=1.0,sigma_w=100.0
#mintest=1.1185098593919596,mintrain=1.1271459168632074,epsw=0.1,epsU=1.0e-10,signal_var=1.0,sigma_w=10.0
#mintest=1.117989155962634,mintrain=1.1265498526975748,epsw=10.0,epsU=1.0e-12,signal_var=1.0,sigma_w=10.0
#mintest=1.1186600606286057,mintrain=1.1273938704339113,epsw=1.0,epsU=1.0e-9,signal_var=1.0,sigma_w=1.0

