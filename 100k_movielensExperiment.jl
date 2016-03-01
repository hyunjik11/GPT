@everywhere using DataFrames
@everywhere using GPT_SGLD
@everywhere using Distributions

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
@everywhere n1,D1=size(UserData); n2,D2=size(MovieData); 
@everywhere ytrain = Rating[1:Ntrain,3];
@everywhere ytest = Rating[Ntrain+1:Ntrain+Ntest,3];
@everywhere ytrainMean=mean(ytrain);
@everywhere ytrainStd=std(ytrain);
@everywhere ytrain=datawhitening(ytrain);
@everywhere ytest = (ytest-ytrainMean)/ytrainStd;
@everywhere Ratingtrain=hcat(Rating[1:Ntrain,1:2],ytrain);
@everywhere Ratingtest=hcat(Rating[Ntrain+1:Ntrain+Ntest,1:2],ytest);
@everywhere n = 30; 
@everywhere M = 5;
@everywhere burnin=0;
@everywhere numiter=30;
@everywhere r = 8
@everywhere Q=r;   
@everywhere D = 2;
@everywhere signal_var = 0.001;
@everywhere param_seed=17;
@everywhere I=repmat(1:r,1,2);
@everywhere m = 100;
@everywhere maxepoch = 20;
@everywhere epsw=1e-2
@everywhere epsU=1e-10
@everywhere sigma_w=sqrt(n1*n2)/r

@everywhere numbatches=int(ceil(maximum(Ntrain)/m));
@everywhere a=1;b1=1;b2=2;
L=100;
norm=Array(Float64,L);
UserHashmap=Array(Int64,M,n1); MovieHashmap=Array(Int64,M,n2);
for i=1:n1
	UserHashmap[:,i]=sample(1:n,M,replace=false)
end
for i=1:n2
	MovieHashmap[:,i]=sample(1:n,M,replace=false)
end
UserBHashmap=2*rand(Bernoulli(),M,n1)-1
MovieBHashmap=2*rand(Bernoulli(),M,n2)-1
a=1;b1=1;b2=1;

# function to cutoff predictions at 1 and 5
function cutoff!(pred::Vector)
	idxlow=(pred.<1); pred[idxlow]=1;
	idxhigh=(pred.>5); pred[idxhigh]=5;
end

# function for tensor model for CF with no side information, using full W
function GPT_test(Rating::Array,UserData::Array,MovieData::Array,Ratingtest::Array, signal_var::Real, sigma_w::Real, r::Integer, m::Integer, epsw::Real, epsU::Real,burnin::Integer, maxepoch::Integer, param_seed::Integer; langevin::Bool=true, stiefel::Bool=true)
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
				else U+=epsU*(gradU-n1*U)/2+sqrt(epsU)*randn(n1,r); V+=epsU*(gradV-n2*U)/2+sqrt(epsU)*randn(n2,r);
				end
			else
				if stiefel
				        momU=proj(U,sqrt(epsU)*gradU/2); momV=proj(V,sqrt(epsU)*gradV/2);
				        U=geod(U,momU,sqrt(epsU)); V=geod(V,momV,sqrt(epsU));
				        if U==zeros(n1,r) || V==zeros(n2,r)#if NaN appears while evaluating G
				            return zeros(r,r,maxepoch),zeros(n1,r,maxepoch),zeros(n2,r,maxepoch)
				        end
				else U+=epsU*(gradU-n1*U)/2; V+=epsU*(gradV-n2*U)/2;
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
			
			counter=0;
			for i=1:Ntest
				user=Ratingtest[i,1]; movie=Ratingtest[i,2];
				testpred[i]=(testpred[i]*counter+sum((U[user,:]*w).*V[movie,:]))/(counter+1)
			end
			final_testpred=testpred*ytrainStd+ytrainMean;
			cutoff!(final_testpred);
			testRMSE=sqrt(sum((ytrainStd*Ratingtest[:,3]+ytrainMean-final_testpred).^2)/Ntest)
			#counter+=1
			println("epoch=$epoch, trainRMSE=$trainRMSE, testRMSE=$testRMSE")
		end
	end
	return w_store,U_store,V_store
end
for epsw in [0.08,0.04,0.02,0.01,0.005,0.003], epsU in [8e-10,4e-10,2e-10,1e-10,0.5e-10,0.3e-10]
println("epsw=$epsw , epsU=$epsU")
w_store,U_store,V_store=GPT_test(Ratingtrain,UserData,MovieData,Ratingtest,signal_var, sigma_w, r, m, epsw, epsU, burnin, maxepoch, param_seed);
end

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

