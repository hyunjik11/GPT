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
	q=[1,18,25,35,45,50,56]
	map(x->indmin(q.<x),age)
end


# function to cutoff predictions at 1 and 5
@everywhere function cutoff!(pred::Vector)
	idxlow=(pred.<1); pred[idxlow]=1;
	idxhigh=(pred.>5); pred[idxhigh]=5;
end


# function for tensor model for CF with no side information, using SGD to learn U,V with fixed w.
@everywhere function GPT_fixw(Rating::Array,UserData::Array,MovieData::Array,Ratingtest::Array, signal_var::Real, sigma_u::Real, w::Array, m::Integer, epsU::Real,burnin::Integer, maxepoch::Integer, param_seed::Integer;langevin::Bool=false, stiefel::Bool=false,avg::Bool=false)
	N=size(Rating,1);Ntest=size(Ratingtest,1);
	n1,D1=size(UserData); 
	n2,D2=size(MovieData);
	numbatches=int(ceil(N/m));
	r=size(w,1);
	# initialise U,V
	srand(param_seed);
	U_store=Array(Float64,n1,r,maxepoch)
	V_store=Array(Float64,testRMSE=0.9515487492625331n2,r,maxepoch)
	testpred_store=Array(Float64,Ntest,maxepoch);
	U=sigma_u*randn(n1,r);V=sigma_u*randn(n2,r);

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
			gradU=zeros(n1,r);
			gradV=zeros(n2,r);
			for ii=1:batch_size
				user=batch_ratings[ii,1]; movie=batch_ratings[ii,2]; rating=batch_ratings[ii,3];
				pred=sum((U[user,:]*w).*V[movie,:])
				gradU[user,:]+=(rating-pred)*V[movie,:]*w'/signal_var
				gradV[movie,:]+=(rating-pred)*U[user,:]*w/signal_var
			end
			gradU*=N/batch_size;
			gradV*=N/batch_size;

			# update U,V
			if langevin
				if stiefel
			        momU=proj(U,sqrt(epsU)*gradU/2+randn(n1,r)); momV=proj(V,sqrt(epsU)*gradV/2+randn(n2,r));
			        U=geod(U,momU,sqrt(epsU)); V=geod(V,momV,sqrt(epsU));
			        if U==zeros(n1,r) || V==zeros(n2,r)#if NaN appears while evaluating G
			            return zeros(r,r,maxepoch),zeros(n1,r,maxepoch),zeros(n2,r,maxepoch)
			        end
				else U+=epsU*(gradU-U/sigma_u^2)/2+sqrt(epsU)*randn(n1,r); V+=epsU*(gradV-V/sigma_u^2)/2+sqrt(epsU)*randn(n2,r);
				end
			else
				if stiefel
			        momU=proj(U,sqrt(epsU)*gradU/2); momV=proj(V,sqrt(epsU)*gradV/2);
			        U=geod(U,momU,sqrt(epsU)); V=geod(V,momV,sqrt(epsU));
			        if U==zeros(n1,r) || V==zeros(n2,r)#if NaN appears while evaluating G
			            return zeros(r,r,maxepoch),zeros(n1,r,maxepoch),zeros(n2,r,maxepoch)
			        end
				else U+=epsU*(gradU-U/sigma_u^2)/2; V+=epsU*(gradV-V/sigma_u^2)/2;
				end
			end
		end
		
		if epoch>burnin
			U_store[:,:,epoch-burnin]=U
			V_store[:,:,epoch-burnin]=V
			if ~avg
				counter=0;
			end
			trainpred=Array(Float64,N)
			for i=1:N
				user=Rating[i,1]; movie=Rating[i,2]; 
				trainpred[i]=(trainpred[i]*counter+sum((U[user,:]*w).*V[movie,:]))/(counter+1)
			end
			final_trainpred=trainpred*ytrainStd+ytrainMean;
			cutoff!(final_trainpred);
			trainRMSE=sqrt(sum((ytrainStd*Rating[:,3]+ytrainMean-final_trainpred).^2)/N)
			trainRMSEvec[epoch-burnin]=trainRMSE
			
			for i=1:Ntest
				user=Ratingtest[i,1]; movie=Ratingtest[i,2];
				testpred[i]=(testpred[i]*counter+sum((U[user,:]*w).*V[movie,:]))/(counter+1)
			end
			final_testpred=testpred*ytrainStd+ytrainMean;
			cutoff!(final_testpred);
			testpred_store[:,epoch-burnin]=final_testpred;
			testRMSE=sqrt(sum((ytrainStd*Ratingtest[:,3]+ytrainMean-final_testpred).^2)/Ntest)
			testRMSEvec[epoch-burnin]=testRMSE			
			counter+=1
			println("epoch=$epoch, trainRMSE=$trainRMSE, testRMSE=$testRMSE")
		end
	end
	return U_store,V_store,testpred_store,trainRMSEvec,testRMSEvec
end

# function for tensor model for CF with no side information, using fixed w and Gibbs sampling for U,V
@everywhere function GPT_fixw_gibbs(Rating::Array,UserData::Array,MovieData::Array,Ratingtest::Array, signal_var::Real, sigma_u::Real,w::Array, burnin::Integer, maxepoch::Integer, n_samples::Integer, param_seed::Integer;avg::Bool=false,rotated_w::Bool=false)
	println("GPT_fixw_gibbs, signal_var=$signal_var,sigma_u=$sigma_u")
	N=size(Rating,1);Ntest=size(Ratingtest,1);
	n1,D1=size(UserData); 
	n2,D2=size(MovieData);
	r=size(w,1);
	
	# initialise w,U,V
	srand(param_seed);
	U_store=Array(Float64,n1,r,maxepoch)
	V_store=Array(Float64,n2,r,maxepoch)
	testpred_store=Array(Float64,Ntest,maxepoch);
	Q,R=qr(randn(r,r))
	U=sigma_u*randn(n1,r);V=sigma_u*randn(n2,r);
	if rotated_w		
		w=Q*w;
		U*=Q'
	end

	trainRMSEvec=Array(Float64,maxepoch)
	testRMSEvec=Array(Float64,maxepoch)
	testpred=zeros(Ntest)
	counter=0;
	for epoch=1:(burnin+maxepoch)
		for gibbs=1:n_samples		
			# update U
			for i=1:n1
				idx=(Rating[:,1].==i) #BitArray size N
				if 	sum(idx)>0		
					Ni=Rating[idx,2] #vector of movies rated by user i
					yi=Rating[idx,3] #vector of these ratings
					VNiw=V[Ni,:]*w';
					invSigmai=VNiw'*VNiw/signal_var+eye(r)/(sigma_u^2);
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
					invSigmaj=UNjw'*UNjw/signal_var+eye(r)/(sigma_u^2);
					muj=\(invSigmaj,UNjw'*yj)/signal_var;
					V[j,:]=\(chol(invSigmaj,:U),randn(r))+muj
				#else println("movie $j has not been rated in training set")
				end
			end
		end
		if epoch>burnin
			U_store[:,:,epoch-burnin]=U
			V_store[:,:,epoch-burnin]=V
			if ~avg			
				counter=0;
			end
			trainpred=Array(Float64,N)
			for i=1:N
				user=Rating[i,1]; movie=Rating[i,2]; 
				trainpred[i]=(trainpred[i]*counter+sum((U[user,:]*w).*V[movie,:]))/(counter+1)
			end
			final_trainpred=trainpred*ytrainStd+ytrainMean;
			cutoff!(final_trainpred);
			trainRMSE=sqrt(sum((ytrainStd*Rating[:,3]+ytrainMean-final_trainpred).^2)/N)
			trainRMSEvec[epoch-burnin]=trainRMSE
			
			for i=1:Ntest
				user=Ratingtest[i,1]; movie=Ratingtest[i,2];
				testpred[i]=(testpred[i]*counter+sum((U[user,:]*w).*V[movie,:]))/(counter+1)
			end
			final_testpred=testpred*ytrainStd+ytrainMean;
			cutoff!(final_testpred);
			testpred_store[:,epoch-burnin]=final_testpred;
			testRMSE=sqrt(sum((ytrainStd*Ratingtest[:,3]+ytrainMean-final_testpred).^2)/Ntest)
			testRMSEvec[epoch-burnin]=testRMSE
			counter+=1
			println("epoch=$epoch, trainRMSE=$trainRMSE, testRMSE=$testRMSE")
		end
	end
	return U_store,V_store,testpred_store,trainRMSEvec,testRMSEvec
end

# function for tensor model for CF with no side information, using full w and SGD to learn U,V,W
# use U,V~N(0,1), but note we'll need different step sizes for stiefel
@everywhere function GPT_fullw(Rating::Array,UserData::Array,MovieData::Array,Ratingtest::Array, signal_var::Real, sigma_u::Real,sigma_w::Real, w_init::Array, m::Integer, epsw::Real, epsU::Real,burnin::Integer, maxepoch::Integer, param_seed::Integer; langevin::Bool=false, stiefel::Bool=false,avg::Bool=false)
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
	testpred_store=Array(Float64,Ntest,maxepoch);
	w=w_init;
	#sigma_w=sqrt(sum(w.^2))/r;
	if stiefel
		Z1=randn(r,n1);	Z2=randn(r,n2)
		U=transpose(\(sqrtm(Z1*Z1'),Z1))
		V=transpose(\(sqrtm(Z2*Z2'),Z2))
    else U=randn(n1,r);V=randn(n2,r);
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
				gradw[:]+=(rating-pred)*kron(V[movie,:]',U[user,:]')/signal_var
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
				else U+=epsU*(gradU-U/sigma_u^2)/2+sqrt(epsU)*randn(n1,r); V+=epsU*(gradV-V/sigma_u^2)/2+sqrt(epsU)*randn(n2,r);
				end
			else
				if stiefel
				        momU=proj(U,sqrt(epsU)*gradU/2); momV=proj(V,sqrt(epsU)*gradV/2);
				        U=geod(U,momU,sqrt(epsU)); V=geod(V,momV,sqrt(epsU));
				        if U==zeros(n1,r) || V==zeros(n2,r)#if NaN appears while evaluating G
				            return zeros(r,r,maxepoch),zeros(n1,r,maxepoch),zeros(n2,r,maxepoch)
				        end
				else U+=epsU*(gradU-U/sigma_u^2)/2; V+=epsU*(gradV-V/sigma_u^2)/2;
				end
			end
		end
		
		if epoch>burnin
			w_store[:,:,epoch-burnin]=w
			U_store[:,:,epoch-burnin]=U
			V_store[:,:,epoch-burnin]=V
			if ~avg			
				counter=0;
			end
			trainpred=Array(Float64,N)
			for i=1:N
				user=Rating[i,1]; movie=Rating[i,2]; 
				trainpred[i]=(trainpred[i]*counter+sum((U[user,:]*w).*V[movie,:]))/(counter+1)
			end
			final_trainpred=trainpred*ytrainStd+ytrainMean;
			cutoff!(final_trainpred);
			trainRMSE=sqrt(sum((ytrainStd*Rating[:,3]+ytrainMean-final_trainpred).^2)/N)
			trainRMSEvec[epoch-burnin]=trainRMSE
			
			for i=1:Ntest
				user=Ratingtest[i,1]; movie=Ratingtest[i,2];
				testpred[i]=(testpred[i]*counter+sum((U[user,:]*w).*V[movie,:]))/(counter+1)
			end
			final_testpred=testpred*ytrainStd+ytrainMean;
			cutoff!(final_testpred);
			testpred_store[:,epoch-burnin]=final_testpred;
			testRMSE=sqrt(sum((ytrainStd*Ratingtest[:,3]+ytrainMean-final_testpred).^2)/Ntest)
			testRMSEvec[epoch-burnin]=testRMSE
			counter+=1
			println("epoch=$epoch, trainRMSE=$trainRMSE, testRMSE=$testRMSE")
		end
	end
	return w_store,U_store,V_store,testpred_store,trainRMSEvec,testRMSEvec
end

# function for tensor model for CF with no side information, using full w and Gibbs sampling for U,V,w
@everywhere function GPT_fullw_gibbs(Rating::Array,UserData::Array,MovieData::Array,Ratingtest::Array, signal_var::Real, sigma_u::Real,sigma_w::Real,w_init::Array, burnin::Integer, maxepoch::Integer, n_samples::Integer, param_seed::Integer;avg::Bool=false,rotated_w::Bool=false)
	println("GPT_fullw_gibbs, signal_var=$signal_var,sigma_u=$sigma_u")
	N=size(Rating,1);Ntest=size(Ratingtest,1);
	n1,D1=size(UserData); 
	n2,D2=size(MovieData);
	r=size(w_init,1);
	
	# initialise w,U,V
	srand(param_seed);
	w_store=Array(Float64,r,r,maxepoch)
	U_store=Array(Float64,n1,r,maxepoch)
	V_store=Array(Float64,n2,r,maxepoch)	
	testpred_store=Array(Float64,Ntest,maxepoch);
	Q,R=qr(randn(r,r))
	U=sigma_u*randn(n1,r);V=sigma_u*randn(n2,r);
	if rotated_w		
		w=Q*w_init;
		U*=Q';
	else w=w_init;
	end
	#sigma_w=sqrt(sum(w.^2))/r;

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
					invSigmai=VNiw'*VNiw/signal_var+eye(r)/(sigma_u^2);
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
					invSigmaj=UNjw'*UNjw/signal_var+eye(r)/(sigma_u^2);
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
			if ~avg			
				counter=0;
			end
			trainpred=Array(Float64,N)
			for i=1:N
				user=Rating[i,1]; movie=Rating[i,2]; 
				trainpred[i]=(trainpred[i]*counter+sum((U[user,:]*w).*V[movie,:]))/(counter+1)
			end
			final_trainpred=trainpred*ytrainStd+ytrainMean;
			cutoff!(final_trainpred);
			trainRMSE=sqrt(sum((ytrainStd*Rating[:,3]+ytrainMean-final_trainpred).^2)/N)
			trainRMSEvec[epoch-burnin]=trainRMSE
			
			for i=1:Ntest
				user=Ratingtest[i,1]; movie=Ratingtest[i,2];
				testpred[i]=(testpred[i]*counter+sum((U[user,:]*w).*V[movie,:]))/(counter+1)
			end
			final_testpred=testpred*ytrainStd+ytrainMean;
			cutoff!(final_testpred);
			testpred_store[:,epoch-burnin]=final_testpred;
			testRMSE=sqrt(sum((ytrainStd*Ratingtest[:,3]+ytrainMean-final_testpred).^2)/Ntest)
			testRMSEvec[epoch-burnin]=testRMSE
			counter+=1
			println("epoch=$epoch, trainRMSE=$trainRMSE, testRMSE=$testRMSE")
		end
	end
	return w_store,U_store,V_store,testpred_store,trainRMSEvec,testRMSEvec
end

# function for tensor model for CF with side information, using SGD to learn U,V with fixed w.
# only use side info for users with freq <= uft and movies with freq <= mft
@everywhere function GPT_fixw_sideinfo_thresh(Rating::Array,UserData::Array,MovieData::Array,Ratingtest::Array,userfreq::Vector, moviefreq::Vector, signal_var::Real, sigma_u::Real, w::Array, m::Integer, epsU::Real,a::Real,b::Real,c::Real,burnin::Integer, maxepoch::Integer, param_seed::Integer;langevin::Bool=false, stiefel::Bool=false,avg::Bool=false)
	#println("langevin=$langevin")
	N=size(Rating,1);Ntest=size(Ratingtest,1);
	n1,D1=size(UserData); 
	n2,D2=size(MovieData);
	numbatches=int(ceil(N/m));
	r=size(w,1);
	# initialise U,V
	srand(param_seed);
	U_store=Array(Float64,n1+D1,r,maxepoch)
	V_store=Array(Float64,n2+D2,r,maxepoch)
	testpred_store=Array(Float64,Ntest,maxepoch);
	U=sigma_u*randn(n1+D1,r);V=sigma_u*randn(n2+D2,r);

	Uidx=Dict{Int64,Array{Int64}}(); Vidx=Dict{Int64,Array{Int64}}(); #dictionary of feature indices in U/V for each user/movie
	for user=1:n1
		userfeat_ind=find(UserData[user,:]);
		Uidx[user]=n1+userfeat_ind;
	end
	for movie=1:n2
		moviefeat_ind=find(MovieData[movie,:]);
		Vidx[movie]=n2+moviefeat_ind;
	end
	
	prob_user_in_m=Array(Float64,n1); prob_movie_in_m=Array(Float64,n2); #prob(user/movie in arbitrary minibatch). Need for bias correction.
	for user=1:n1
		temp=[1-userfreq[user]/(N-i+1) for i=1:m];
		prob_user_in_m[user]=1-exp(sum(log(temp)));
	end
	for movie=1:n2
		temp=[1-moviefreq[movie]/(N-i+1) for i=1:m];
		prob_movie_in_m[movie]=1-exp(sum(log(temp)));
	end

	trainRMSEvec=Array(Float64,maxepoch)
	testRMSEvec=Array(Float64,maxepoch)
	trainpred=zeros(N)
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
			gradU=zeros(n1+D1,r);
			gradV=zeros(n2+D2,r);
			Uindices=zeros(1+D1,batch_size);Vindices=zeros(1+D2,batch_size); # large enough container for all indices occuring in minibatch
			for ii=1:batch_size
				user=batch_ratings[ii,1]; movie=batch_ratings[ii,2]; rating=batch_ratings[ii,3];
				uidx=Uidx[user]; vidx=Vidx[movie]; #feature rows of U,V to be updated
				lu=length(uidx); lv=length(vidx);
				sumU=U[user,:]+b*sum(U[uidx,:],1); sumV=V[movie,:]+c*sum(V[vidx,:],1); 
				pred=a*sum((sumU*w).*sumV)
				Utemp=(rating-pred)*sumV*w'; Vtemp=(rating-pred)*sumU*w;
				gradU[user,:]+=a*Utemp/signal_var;
				gradV[movie,:]+=a*Vtemp/signal_var;
				gradU[uidx,:]+=a*b*repmat(Utemp,lu)/signal_var
				gradV[vidx,:]+=a*c*repmat(Vtemp,lv)/signal_var
				Uindices[1:1+lu,ii]=[user,uidx]; Vindices[1:1+lv,ii]=[movie,vidx];
			end
			gradU*=N/batch_size;gradV*=N/batch_size;
			
			# update U,V
			if langevin
				Uindices=sort(union(vec(Uindices)))[2:end];
				Vindices=sort(union(vec(Vindices)))[2:end];
				Unoise=zeros(n1+D1,r); Unoise[Uindices,:]=sqrt(epsU)*randn(length(Uindices),r);
				Vnoise=zeros(n2+D2,r); Vnoise[Vindices,:]=sqrt(epsU)*randn(length(Vindices),r);
				U+=epsU*(gradU-U/sigma_u^2)/2+Unoise
				V+=epsU*(gradV-V/sigma_u^2)/2+Vnoise
				if batch==numbatches
					#println("fnorm(Ugrad)=",epsU*sqrt(sum((gradU-U/sigma_u^2).^2)),",fnorm(Unoise)=",sqrt(sum(Unoise.^2)));
					#println("fnorm(Vgrad)=",epsU*sqrt(sum((gradV-V/sigma_u^2).^2)),",fnorm(Vnoise)=",sqrt(sum(Vnoise.^2)));
				end
			else U+=epsU*(gradU-U/sigma_u^2)/2
				 V+=epsU*(gradV-V/sigma_u^2)/2
			end
			
			#=
			if langevin
				if stiefel
			        momU=proj(U,sqrt(epsU)*gradU/2+randn(n1+D1,r)); momV=proj(V,sqrt(epsU)*gradV/2+randn(n2+D2,r));
			        U=geod(U,momU,sqrt(epsU)); V=geod(V,momV,sqrt(epsU));
			        if U==zeros(n1+D1,r) || V==zeros(n2+D2,r)#if NaN appears while evaluating G
			            return zeros(r,r,maxepoch),zeros(n1+D1,r,maxepoch),zeros(n2+D2,r,maxepoch)
			        end
				else U+=epsU*(gradU-U/sigma_u^2)/2+sqrt(epsU)*randn(n1+D1,r); V+=epsU*(gradV-V/sigma_u^2)/2+sqrt(epsU)*randn(n2+D2,r);
				end
			else
				if stiefel
			        momU=proj(U,sqrt(epsU)*gradU/2); momV=proj(V,sqrt(epsU)*gradV/2);
			        U=geod(U,momU,sqrt(epsU)); V=geod(V,momV,sqrt(epsU));
			        if U==zeros(n1+D1,r) || V==zeros(n2+D2,r)#if NaN appears while evaluating G
			            return zeros(r,r,maxepoch),zeros(n1+D1,r,maxepoch),zeros(n2+D2,r,maxepoch)
			        end
				else U+=epsU*(gradU-U/sigma_u^2)/2; V+=epsU*(gradV-V/sigma_u^2)/2;
				end
			end
			=#
		end
		
		if epoch>burnin
			U_store[:,:,epoch-burnin]=U
			V_store[:,:,epoch-burnin]=V
			if ~avg
				counter=0;
			end
			
			for i=1:N
				user=Rating[i,1]; movie=Rating[i,2];
				uidx=Uidx[user]; vidx=Vidx[movie]; #relevant rows of U,V to be updated
				sumU=U[user,:]+b*sum(U[uidx,:],1); sumV=V[movie,:]+c*sum(V[vidx,:],1);
				trainpred[i]=(trainpred[i]*counter+a*sum((sumU*w).*sumV))/(counter+1)
			end
			final_trainpred=trainpred*ytrainStd+ytrainMean;
			cutoff!(final_trainpred);
			trainRMSE=sqrt(sum((ytrainStd*Rating[:,3]+ytrainMean-final_trainpred).^2)/N)
			trainRMSEvec[epoch-burnin]=trainRMSE
			
			for i=1:Ntest
				user=Ratingtest[i,1]; movie=Ratingtest[i,2];
				uidx=Uidx[user]; vidx=Vidx[movie]; #relevant rows of U,V to be updated
				sumU=U[user,:]+b*sum(U[uidx,:],1); sumV=V[movie,:]+c*sum(V[vidx,:],1);
				testpred[i]=(testpred[i]*counter+a*sum((sumU*w).*sumV))/(counter+1)
			end
			final_testpred=testpred*ytrainStd+ytrainMean;
			cutoff!(final_testpred);
			testpred_store[:,epoch-burnin]=final_testpred;
			testRMSE=sqrt(sum((ytrainStd*Ratingtest[:,3]+ytrainMean-final_testpred).^2)/Ntest)
			testRMSEvec[epoch-burnin]=testRMSE			
			counter+=1
			#println("epoch=$epoch, trainRMSE=$trainRMSE, testRMSE=$testRMSE")
		end
	end
	return U_store,V_store,testpred_store,trainRMSEvec,testRMSEvec
end

# function for tensor model for CF with side information, using SGD to learn U,V with fixed w.
@everywhere function GPT_fixw_sideinfo(Rating::Array,UserData::Array,MovieData::Array,Ratingtest::Array, signal_var::Real, sigma_u::Real, w::Array, m::Integer, epsU::Real,a::Real,b::Real,c::Real,burnin::Integer, maxepoch::Integer, param_seed::Integer;langevin::Bool=false, stiefel::Bool=false,avg::Bool=false)
	#println("langevin=$langevin")
	N=size(Rating,1);Ntest=size(Ratingtest,1);
	n1,D1=size(UserData); 
	n2,D2=size(MovieData);
	numbatches=int(ceil(N/m));
	r=size(w,1);
	# initialise U,V
	srand(param_seed);
	U_store=Array(Float64,n1+D1,r,maxepoch)
	V_store=Array(Float64,n2+D2,r,maxepoch)
	testpred_store=Array(Float64,Ntest,maxepoch);
	U=sigma_u*randn(n1+D1,r);V=sigma_u*randn(n2+D2,r);

	Uidx=Dict{Int64,Array{Int64}}(); Vidx=Dict{Int64,Array{Int64}}(); #dictionary of feature indices in U/V for each user/movie
	for user=1:n1
		userfeat_ind=find(UserData[user,:]);
		Uidx[user]=n1+userfeat_ind;
	end
	for movie=1:n2
		moviefeat_ind=find(MovieData[movie,:]);
		Vidx[movie]=n2+moviefeat_ind;
	end

	trainRMSEvec=zeros(maxepoch)
	testRMSEvec=zeros(maxepoch)
	trainpred=zeros(N)
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
			gradU=zeros(n1+D1,r);
			gradV=zeros(n2+D2,r);
			for ii=1:batch_size
				user=batch_ratings[ii,1]; movie=batch_ratings[ii,2]; rating=batch_ratings[ii,3];
				uidx=Uidx[user]; vidx=Vidx[movie]; #feature rows of U,V to be updated
				lu=length(uidx); lv=length(vidx);
				sumU=U[user,:]+b*sum(U[uidx,:],1); sumV=V[movie,:]+c*sum(V[vidx,:],1); 
				pred=a*sum((sumU*w).*sumV)
				Utemp=(rating-pred)*sumV*w'; Vtemp=(rating-pred)*sumU*w;
				gradU[user,:]+=a*Utemp/signal_var;
				gradV[movie,:]+=a*Vtemp/signal_var;
				gradU[uidx,:]+=a*b*repmat(Utemp,lu)/signal_var
				gradV[vidx,:]+=a*c*repmat(Vtemp,lv)/signal_var
			end
			gradU*=N/batch_size;gradV*=N/batch_size;
			
			# update U,V

			if langevin
				if stiefel
			        momU=proj(U,sqrt(epsU)*gradU/2+randn(n1+D1,r)); momV=proj(V,sqrt(epsU)*gradV/2+randn(n2+D2,r));
			        U=geod(U,momU,sqrt(epsU)); V=geod(V,momV,sqrt(epsU));
			        if U==zeros(n1+D1,r) || V==zeros(n2+D2,r)#if NaN appears while evaluating G
			            return zeros(n1+D1,r,maxepoch),zeros(n2+D2,r,maxepoch), testpred_store,trainRMSEvec,testRMSEvec
			        end
				else U+=epsU*(gradU-U/sigma_u^2)/2+sqrt(epsU)*randn(n1+D1,r); V+=epsU*(gradV-V/sigma_u^2)/2+sqrt(epsU)*randn(n2+D2,r);
				end
			else
				if stiefel
			        momU=proj(U,sqrt(epsU)*gradU/2); momV=proj(V,sqrt(epsU)*gradV/2);
			        U=geod(U,momU,sqrt(epsU)); V=geod(V,momV,sqrt(epsU));
			        if U==zeros(n1+D1,r) || V==zeros(n2+D2,r)#if NaN appears while evaluating G
			            return zeros(n1+D1,r,maxepoch),zeros(n2+D2,r,maxepoch), testpred_store,trainRMSEvec,testRMSEvec
			        end
				else U+=epsU*(gradU-U/sigma_u^2)/2; V+=epsU*(gradV-V/sigma_u^2)/2;
				end
			end
		end
		
		if epoch>burnin
			U_store[:,:,epoch-burnin]=U
			V_store[:,:,epoch-burnin]=V
			if ~avg
				counter=0;
			end
			
			for i=1:N
				user=Rating[i,1]; movie=Rating[i,2];
				uidx=Uidx[user]; vidx=Vidx[movie]; #relevant rows of U,V to be updated
				sumU=U[user,:]+b*sum(U[uidx,:],1); sumV=V[movie,:]+c*sum(V[vidx,:],1);
				trainpred[i]=(trainpred[i]*counter+a*sum((sumU*w).*sumV))/(counter+1)
			end
			final_trainpred=trainpred*ytrainStd+ytrainMean;
			cutoff!(final_trainpred);
			trainRMSE=sqrt(sum((ytrainStd*Rating[:,3]+ytrainMean-final_trainpred).^2)/N)
			trainRMSEvec[epoch-burnin]=trainRMSE
			
			for i=1:Ntest
				user=Ratingtest[i,1]; movie=Ratingtest[i,2];
				uidx=Uidx[user]; vidx=Vidx[movie]; #relevant rows of U,V to be updated
				sumU=U[user,:]+b*sum(U[uidx,:],1); sumV=V[movie,:]+c*sum(V[vidx,:],1);
				testpred[i]=(testpred[i]*counter+a*sum((sumU*w).*sumV))/(counter+1)
			end
			final_testpred=testpred*ytrainStd+ytrainMean;
			cutoff!(final_testpred);
			testpred_store[:,epoch-burnin]=final_testpred;
			testRMSE=sqrt(sum((ytrainStd*Ratingtest[:,3]+ytrainMean-final_testpred).^2)/Ntest)
			testRMSEvec[epoch-burnin]=testRMSE			
			counter+=1
			#println("epoch=$epoch, trainRMSE=$trainRMSE, testRMSE=$testRMSE")
		end
	end
	return U_store,V_store,testpred_store,trainRMSEvec,testRMSEvec
end

#return vector of frequency of ratings for each user and each movie
@everywhere function freq(Rating::Array,UserData::Array,MovieData::Array)
	n1=size(UserData,1); n2=size(MovieData,1);
	R=zeros(n1,n2);
	for i=1:size(Rating,1)
		R[Rating[i,1],Rating[i,2]]=Rating[i,3]
	end
	userfreq=Array(Int32,n1); moviefreq=Array(Int32,n2);
	for user=1:n1
		userfreq[user]=countnz(R[user,:])
	end
	for movie=1:n2
		moviefreq[movie]=countnz(R[:,movie])
	end
	return userfreq,moviefreq
end

# function for tensor model for CF with side information, using full w and SGD to learn U,V,W
# either U,V~N(0,1) or Stiefel, but note we'll need different step sizes for stiefel
@everywhere function GPT_fullw_sideinfo(Rating::Array,UserData::Array,MovieData::Array,Ratingtest::Array, signal_var::Real, sigma_u::Real,sigma_w::Real,w_init::Array, m::Integer, epsw::Real, epsU::Real,a::Real,b::Real,c::Real,burnin::Integer, maxepoch::Integer, param_seed::Integer; langevin::Bool=false, stiefel::Bool=false,avg::Bool=false)
	N=size(Rating,1);Ntest=size(Ratingtest,1);
	n1,D1=size(UserData); 
	n2,D2=size(MovieData);
	numbatches=int(ceil(N/m));
	r=size(w_init,1);
	
	# initialise w,U,V
	srand(param_seed);
	w_store=Array(Float64,r,r,maxepoch)
	U_store=Array(Float64,n1+D1,r,maxepoch)
	V_store=Array(Float64,n2+D2,r,maxepoch)
	testpred_store=Array(Float64,Ntest,maxepoch);
	w=w_init;

	if stiefel
		Z1=randn(r,n1+D1);	Z2=randn(r,n2+D2)
		U=transpose(\(sqrtm(Z1*Z1'),Z1))
		V=transpose(\(sqrtm(Z2*Z2'),Z2))
    else U=sigma_u*randn(n1+D1,r);V=sigma_u*randn(n2+D2,r);
    end
	
	Uidx=Dict{Int64,Array{Int64}}(); Vidx=Dict{Int64,Array{Int64}}(); #dictionary of feature indices in U/V for each user/movie
	for user=1:n1
		userfeat_ind=find(UserData[user,:]);
		Uidx[user]=n1+userfeat_ind;
	end
	for movie=1:n2
		moviefeat_ind=find(MovieData[movie,:]);
		Vidx[movie]=n2+moviefeat_ind;
	end

	trainRMSEvec=Array(Float64,maxepoch)
	testRMSEvec=10*ones(maxepoch)
	trainpred=zeros(N)
	testpred=zeros(Ntest)
	counter=0;
	testcounter=0;
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
			gradU=zeros(n1+D1,r);
			gradV=zeros(n2+D2,r);
			for ii=1:batch_size
				user=batch_ratings[ii,1]; movie=batch_ratings[ii,2]; rating=batch_ratings[ii,3];
				uidx=Uidx[user]; vidx=Vidx[movie]; #feature rows of U,V to be updated
				lu=length(uidx); lv=length(vidx);
				sumU=U[user,:]+b*sum(U[uidx,:],1); sumV=V[movie,:]+c*sum(V[vidx,:],1); 
				pred=a*sum((sumU*w).*sumV)
				Utemp=(rating-pred)*sumV*w'; Vtemp=(rating-pred)*sumU*w;
				gradw[:]+=(rating-pred)*kron((V[movie,:]+c*sum(V[vidx,:],1))',(U[user,:]+b*sum(U[uidx,:],1))')/signal_var
				gradU[user,:]+=a*Utemp/signal_var;
				gradV[movie,:]+=a*Vtemp/signal_var;
				gradU[uidx,:]+=a*b*repmat(Utemp,lu)/signal_var
				gradV[vidx,:]+=a*c*repmat(Vtemp,lv)/signal_var
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
				        momU=proj(U,sqrt(epsU)*gradU/2+randn(n1+D1,r)); momV=proj(V,sqrt(epsU)*gradV/2+randn(n2+D2,r));
				        U=geod(U,momU,sqrt(epsU)); V=geod(V,momV,sqrt(epsU));
				        if U==zeros(n1+D1,r) || V==zeros(n2+D2,r)#if NaN appears while evaluating G
				            return zeros(r,r,maxepoch),zeros(n1+D1,r,maxepoch),zeros(n2+D2,r,maxepoch),testpred_store,trainRMSEvec,testRMSEvec
				        end
				else U+=epsU*(gradU-U/sigma_u^2)/2+sqrt(epsU)*randn(n1+D1,r); V+=epsU*(gradV-V/sigma_u^2)/2+sqrt(epsU)*randn(n2+D2,r);
				end
			else
				if stiefel
				        momU=proj(U,sqrt(epsU)*gradU/2); momV=proj(V,sqrt(epsU)*gradV/2);
				        U=geod(U,momU,sqrt(epsU)); V=geod(V,momV,sqrt(epsU));
				        if U==zeros(n1+D1,r) || V==zeros(n2+D2,r)#if NaN appears while evaluating G
				            return zeros(r,r,maxepoch),zeros(n1+D1,r,maxepoch),zeros(n2+D2,r,maxepoch),testpred_store,trainRMSEvec,testRMSEvec
				        end
				else U+=epsU*(gradU-U/sigma_u^2)/2; V+=epsU*(gradV-V/sigma_u^2)/2;
				end
			end
		end
		
		if epoch>burnin
			w_store[:,:,epoch-burnin]=w
			U_store[:,:,epoch-burnin]=U
			V_store[:,:,epoch-burnin]=V
			if ~avg			
				counter=0;
			end
			#=
			for i=1:N
				user=Rating[i,1]; movie=Rating[i,2];
				uidx=Uidx[user]; vidx=Vidx[movie]; #relevant rows of U,V to be updated
				sumU=U[user,:]+b*sum(U[uidx,:],1); sumV=V[movie,:]+c*sum(V[vidx,:],1);
				trainpred[i]=(trainpred[i]*counter+a*sum((sumU*w).*sumV))/(counter+1)
			end
			final_trainpred=trainpred*ytrainStd+ytrainMean;
			cutoff!(final_trainpred);
			trainRMSE=sqrt(sum((ytrainStd*Rating[:,3]+ytrainMean-final_trainpred).^2)/N)
			trainRMSEvec[epoch-burnin]=trainRMSE
			=#
			for i=1:Ntest
				user=Ratingtest[i,1]; movie=Ratingtest[i,2];
				uidx=Uidx[user]; vidx=Vidx[movie]; #relevant rows of U,V to be updated
				sumU=U[user,:]+b*sum(U[uidx,:],1); sumV=V[movie,:]+c*sum(V[vidx,:],1);
				testpred[i]=(testpred[i]*counter+a*sum((sumU*w).*sumV))/(counter+1)
			end
			final_testpred=testpred*ytrainStd+ytrainMean;
			cutoff!(final_testpred);
			testpred_store[:,epoch-burnin]=final_testpred;
			testRMSE=sqrt(sum((ytrainStd*Ratingtest[:,3]+ytrainMean-final_testpred).^2)/Ntest)
			testRMSEvec[epoch-burnin]=testRMSE;
			counter+=1
			#println("epoch=$epoch, trainRMSE=$trainRMSE, testRMSE=$testRMSE")
			println("epoch=$epoch, testRMSE=$testRMSE")
			if epoch>1 && testRMSE>testRMSEvec[epoch-burnin-1]
				testcounter+=1
			else testcounter=0
			end
		end
		if testcounter>=5 #if testRMSE has increased at least 5 times in a row
			return w_store,U_store,V_store,testpred_store,trainRMSEvec,testRMSEvec
		end
	end
	return w_store,U_store,V_store,testpred_store,trainRMSEvec,testRMSEvec
end

#return vector of freq vs RMSE for users and for movies
@everywhere function RMSEbyfreq(userfreq::Vector,moviefreq::Vector,Rating::Array,pred::Array)
	n1=length(userfreq); n2=length(moviefreq);
	userfrequencies=sort(union(userfreq)); moviefrequencies=sort(union(moviefreq));
	uRMSE=Array(Float64,length(userfrequencies)); mRMSE=Array(Float64,length(moviefrequencies));
	for i=1:length(userfrequencies)
		freq=userfrequencies[i];
		users=find(userfreq.==freq);
		indices=findin(Rating[:,1],users);
		uRMSE[i]=sqrt(sum((Rating[indices,3]-pred[indices]).^2)/length(indices))
	end
	for j=1:length(moviefrequencies)
		freq=moviefrequencies[j];
		movies=find(moviefreq.==freq);
		indices=findin(Rating[:,2],movies);
		mRMSE[j]=sqrt(sum((Rating[indices,3]-pred[indices]).^2)/length(indices))
	end
	return userfrequencies,moviefrequencies,uRMSE,mRMSE
end

#return vector of freq vs RMSE for users and for movies
@everywhere function RMSEbyfreq_binned(userfreq::Vector,moviefreq::Vector,Rating::Array,pred::Array,nbins::Integer)
	qu=linspace(0,maximum(userfreq),nbins+1)[2:end]
	map(x->indmin(qu.<x),userfreq);
	qm=linspace(0,maximum(moviefreq),nbins+1)[2:end]
	map(x->indmin(qm.<x),moviefreq);
	uRMSE=Array(Float64,nbins); mRMSE=Array(Float64,nbins);
	for i=1:nbins
		users=find(userfreq.==i);
		indices=findin(Rating[:,1],users);
		uRMSE[i]=sqrt(sum((Rating[indices,3]-pred[indices]).^2)/length(indices))
		movies=find(moviefreq.==i);
		indices=findin(Rating[:,2],movies);
		mRMSE[i]=sqrt(sum((Rating[indices,3]-pred[indices]).^2)/length(indices))
	end
	return qu,qm,uRMSE,mRMSE
end

@everywhere function bin_movie(genre::Array) #genre is an array where each row has the genres of film as string. Some entries can be empty strings
	genre_list=["Action","Adventure","Animation","Children's","Comedy","Crime","Documentary","Drama","Fantasy","Film-Noir","Horror","Musical","Mystery","Romance","Sci-Fi","Thriller","War","Western"];
	n2,D=size(genre);
	dummy_genre=zeros(n2,length(genre_list)); 
	for i=1:n2
		ind=findin(genre_list,genre[i,:]);
		dummy_genre[i,ind]=1;
	end
	return dummy_genre
end
	

##data clearing
@everywhere UserData = readdlm("ml-1m/users.txt", '|');
@everywhere MovieData = readdlm("ml-1m/movies.txt",'|');
@everywhere Rating = readdlm("ml-1m/ratings.txt",'|',Float64)[:,1:end-1];

@everywhere N=size(Rating,1);
@everywhere Ntrain = 800000;
@everywhere Ntest = N-Ntrain;
@everywhere UserData[:,3] = bin_age(UserData[:,3])
@everywhere UserData = convertdummy(convert(DataFrame,UserData),[:x2,:x3,:x4])[:,1:end-1];
@everywhere MovieIdx=MovieData[:,1]; #"Some MovieIDs do not correspond to a movie due to accidental duplicate entries and/or test entries" 
@everywhere MovieData = bin_movie(MovieData[:,3:end]);
@everywhere UserData = convert(Array{Float64,2},UserData)[:,2:end];
@everywhere MovieData = convert(Array{Float64,2},MovieData);
@everywhere Rating[:,2]=indexin(Rating[:,2],MovieIdx); #adjust movieIDs, mapping them back onto 
@everywhere n1,D1=size(UserData); 
@everywhere n2,D2=size(MovieData); 
@everywhere n = 30; 
@everywhere M = 5;
@everywhere numiter=30;
@everywhere r = 15;
@everywhere Q=r;   
@everywhere D = 2;
@everywhere signal_var = 0.8;
@everywhere m = 1000;

@everywhere numbatches=int(ceil(maximum(Ntrain)/m));
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

@everywhere param_seed=17;
@everywhere a=0.5;
@everywhere b=0.25;
@everywhere c=0.5;
@everywhere burnin=0;
@everywhere maxepoch = 1;
@everywhere epsw=1e-4;
@everywhere epsU=1e-6;
@everywhere sigma_w=1;
@everywhere sigma_u=0.1;
@everywhere signal_var=0.8;
@everywhere w_init=sigma_w*randn(r,r);
#signal_var=0.5; sigma_u=0.5; sigma_w=sqrt(sum(w_init.^2))/r; epsw=0.3; epsU=1e-4; burnin=15; maxepoch=1000; #n_samples=8; m=100; a=0.25;b=0.25;c=0.3; param_seed=10;
@everywhere userfreq,moviefreq=freq(Ratingtrain,UserData,MovieData);
@everywhere seed=1;
@everywhere srand(seed);
@everywhere perm=Array(Float64,N,5);
@everywhere for i=1:5
	perm[:,i]=randperm(N);
end
myRating=Rating[perm[:,split],:];
		ytrain = myRating[1:Ntrain,3];
		ytest = myRating[Ntrain+1:Ntrain+Ntest,3];
		ytrainMean=mean(ytrain);
		ytrainStd=std(ytrain);
		ytrain=(ytrain-ytrainMean)/ytrainStd;
		ytest = (ytest-ytrainMean)/ytrainStd;
		Ratingtrain=hcat(myRating[1:Ntrain,1:2],ytrain);
		Ratingtest=hcat(myRating[Ntrain+1:Ntrain+Ntest,1:2],ytest);
		w_store,U_store,V_store,testpred_store,trainRMSEvec,testRMSEvec=GPT_fullw_sideinfo(Ratingtrain,UserData,MovieData,Ratingvalid,signal_var,sigma_u,sigma_w,w_init, m, epsw, epsU,a,b,c,burnin, maxepoch, param_seed,langevin=false, stiefel=false,avg=false);
#=
@everywhere t=Iterators.product([1e-3,1e-4,1e-5],[1e-5,1e-6])
@everywhere myt=Array(Any,);
@everywhere it=1;
@everywhere for prod in t
	myt[it]=prod;
	    it+=1;
end
@parallel for  Tuple in myt 
	epsw,epsU=Tuple;
	mintest=Array(Float64,5);
	for split=1:5
		myRating=Rating[perm[:,split],:];
		ytrain = myRating[1:Ntrain,3];
		ytest = myRating[Ntrain+1:Ntrain+Ntest,3];
		ytrainMean=mean(ytrain);
		ytrainStd=std(ytrain);
		ytrain=(ytrain-ytrainMean)/ytrainStd;
		ytest = (ytest-ytrainMean)/ytrainStd;
		Ratingtrain=hcat(myRating[1:Ntrain,1:2],ytrain);
		Ratingtest=hcat(myRating[Ntrain+1:Ntrain+Ntest,1:2],ytest);
		w_store,U_store,V_store,testpred_store,trainRMSEvec,testRMSEvec=GPT_fullw_sideinfo(Ratingtrain,UserData,MovieData,Ratingvalid,signal_var,sigma_u,sigma_w,w_init, m, epsw, epsU,a,b,c,burnin, maxepoch, param_seed,langevin=false, stiefel=false,avg=false);
		argmin=indmin(testRMSEvec); mintest[split]=testRMSEvec[argmin];
	end
	mean_mintest=mean(mintest); std_mintest=std(mintest);
	println("mean_mintest=$mintest,std_mintest=$std_mintest,a=$a,b=$b,c=$c,epsw=$epsw,epsU=$epsU,signal_var=$signal_var")
end
=#

