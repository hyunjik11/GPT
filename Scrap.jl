#=
for i=1:10
    subplot(2,5,i)
    ax=gca()
    plot(vec(theta_store[10*i,:]))
    plot(vec(theta_store2[10*i,:]))
    ylim(-5,5)
    xlabel("num_iterations(no burnin)")
    setp(ax[:get_xticklabels](),fontsize=8)
end
=#

# function for tensor model for CF with side information, using full w and SGD to learn U,V,W
# either U,V~N(0,1) or Stiefel, but note we'll need different step sizes for stiefel
@everywhere function GPT_fullw_sideinfo(Rating::Array,UserData::Array,MovieData::Array,Ratingtest::Array, signal_var::Real, sigma_u::Real,sigma_w::Real, w_init::Array, m::Integer, epsw::Real, epsU::Real,a::Real,b::Real,c::Real,burnin::Integer, maxepoch::Integer, param_seed::Integer; langevin::Bool=false, stiefel::Bool=false,avg::Bool=false)
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
	#sigma_w=sqrt(sum(w.^2))/r;
	if stiefel
		Z1=randn(r,n1);	Z2=randn(r,n2)
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
			gradw=zeros(r,r);
			gradU=zeros(n1,r);
			gradV=zeros(n2,r);
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
			println("epoch=$epoch, trainRMSE=$trainRMSE, testRMSE=$testRMSE")
		end
	end
	return w_store,U_store,V_store,testpred_store,trainRMSEvec,testRMSEvec
end

		

