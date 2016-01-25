module GPT_SGLD

using Distributions,Optim,ForwardDiff

export proj, geod, datawhitening, feature, feature2, featureNotensor, gradfeatureNotensor, samplenz, pred, RMSE, parallelRMSE, createmesh,fhatdraw, GPT_SGLDERM, GPT_SGDERM, GPNT_SGLD,GPNT_logmarginal,GPNT_hyperparameters,GPT_GMC,GPT_SGLDERMw,GPT_SGLDERM_RMSprop,GPNT_hyperparameters_ng
    
# define proj for Stiefel manifold
function proj(U::Array,V::Array)
    return V-U*(U'*V+V'*U)/2
end

# define geod for Stiefel manifold - just want endpt
function geod(U::Array,mom::Array,t::Real)
    n,r=size(U)
    A=U'*mom
    temp=[A -mom'*mom;eye(r) A]
    E=expm(t*temp) #can become NaN when temp too large. Return 0 in this case with warning
    if sum(isnan(E))>0
        println("Get NaN when moving along Geodesic. Try smaller epsU") 
        return zeros(n,r)
    else
        mexp=expm(-t*A)
        tmpU=[U mom]*E[:,1:r]*mexp;
        #ensure that tmpU has cols of unit norm
        normconst=Array(Float64,1,r);
        for l=1:r
    	    normconst[1,l]=norm(tmpU[:,l])
        end
        return tmpU./repmat(normconst,n,1)
    end
end

# define geod for Stiefel manifold - want both endpt and mom
function geodboth(U::Array,mom::Array,t::Real)
    n,r=size(U)
    A=U'*mom
    temp=[A -mom'*mom;eye(r) A]
    E=expm(t*temp) #can become NaN when temp too large. Return 0 in this case
    if sum(isnan(E))>0
        println("Get NaN when moving along Geodesic. Try smaller epsU")
        return zeros(n,r),zeros(n,r)
    else
        mexp=expm(-t*A)
        tmpU=[U mom]*E[:,1:r]*mexp;
        tmpV=[U mom]*E[:,(r+1):2r]*mexp;
        #ensure that tmpU has cols of unit norm
        normconst=Array(Float64,1,r);
        for l=1:r
    	    normconst[1,l]=norm(tmpU[:,l])
        end
        return tmpU./repmat(normconst,n,1),tmpV
    end
end

# centre and normalise data X so that each col has sd=1
function datawhitening(X::Array) 
    for i = 1:size(X,2)   
        X[:,i] = (X[:,i] - mean(X[:,i]))/std(X[:,i])   
    end
    return X
end

# extract features from tensor decomp of each row of X using same length scale for each dimension
function feature(X::Array,n::Integer,length_scale::Real, sigma_RBF::Real,seed::Integer,scale::Real)    
    N,D=size(X)
    phi=Array(Float64,n,D,N)
    srand(seed)
    Z=randn(n,D)/length_scale
    b=rand(n,D)*2*pi
    for i=1:N
		for k=1:D
			for j=1:n
				phi[j,k,i]=cos(X[i,k]*Z[j,k]+b[j,k])
			end
		end
    end
    return scale*(sigma_RBF)^(1/D)*sqrt(2/n)*phi
end

# extract features from tensor decomp of each row of X using varying length scales for different dimensions
function feature(X::Array,n::Integer,length_scale::Vector, sigma_RBF::Real,seed::Integer,scale::Real)    
    N,D=size(X)
    if length(length_scale)!=D
		error("dimensions of X and length_scale do not match")
	end
    phi=Array(Float64,n,D,N)
    srand(seed)
    Z=Array(Float64,n,D)
    for k=1:D
		Z[:,k]=randn(n)/length_scale[k]
    end
    b=rand(n,D)*2*pi
    for i=1:N
		for k=1:D
			for j=1:n
				phi[j,k,i]=cos(X[i,k]*Z[j,k]+b[j,k])
			end
		end
    end
    return scale*(sigma_RBF)^(1/D)*sqrt(2/n)*phi
end

# alternative Fourier feature embedding for same length_scale
function feature2(X::Array,n::Integer,length_scale::Real,sigma_RBF::Real,seed::Integer,scale::Real)    
    if n%2==0
		half_n=int(n/2)
		N,D=size(X)
		phi=Array(Float64,n,D,N)
		srand(seed)
		Z=randn(half_n,D)/length_scale
		for i=1:N
			for k=1:D
				for j=1:half_n
					phi[2*j-1,k,i]=sin(X[i,k]*Z[j,k])
					phi[2*j,k,i]=cos(X[i,k]*Z[j,k])
				end
			end
		end
		return scale*(sigma_RBF)^(1/D)*sqrt(2/n)*phi
    else error("n is not even")
    end
end

# alternative Fourier feature embedding for varying length_scales
function feature2(X::Array,n::Integer,length_scale::Vector,sigma_RBF::Real,seed::Integer,scale::Real)    
    if n%2==0
		half_n=int(n/2)
		N,D=size(X)
		if length(length_scale)!=D
			error("dimensions of X and length_scale do not match")
		end
		phi=Array(Float64,n,D,N)
		srand(seed)
		Z=Array(Float64,half_n,D)
		for k=1:D
			Z[:,k]=randn(half_n)/length_scale[k]
		end
		    for i=1:N
				for k=1:D
					for j=1:half_n
						phi[2*j-1,k,i]=sin(X[i,k]*Z[j,k])
						phi[2*j,k,i]=cos(X[i,k]*Z[j,k])
					end
				end
		end
		return scale*(sigma_RBF)^(1/D)*sqrt(2/n)*phi
    else error("n is not even")
    end
end

# fourier feature embedding for the no tensor model (full-theta) using same length_scale
function featureNotensor(X::Array,n::Integer,length_scale::Real,sigma_RBF::Real,seed::Integer)    
    N,D=size(X)
    phi=Array(Float64,n,N)
    srand(seed)
    Z=randn(n,D)/length_scale
    b=2*pi*rand(n)
    for i=1:N
		for j=1:n
        	phi[j,i]=cos(sum(X[i,:].*Z[j,:]) + b[j])
		end
    end
    return sqrt(2/n)*sigma_RBF*phi
end

# fourier feature embedding for the no tensor model (full-theta) using varying length_scales
function featureNotensor(X::Array,n::Integer,length_scale::Vector,sigma_RBF::Real,seed::Integer)    
    N,D=size(X)
	if length(length_scale)!=D
			error("dimensions of X and length_scale do not match")
	end
    phi=Array(Float64,n,N)
    srand(seed)
    Z=Array(Float64,n,D)
    for k=1:D
		Z[:,k]=randn(n)/length_scale[k]
    end
    b=2*pi*rand(n)
    for i=1:N
		for j=1:n
        	phi[j,i]=cos(sum(X[i,:].*Z[j,:]) + b[j])
		end
    end
    return sqrt(2/n)*sigma_RBF*phi
end

# function to give grad of phi wrt length_scale and sigma_RBF. 
# input phi=featureNotensor(X,n,length_scale,sigma_RBF,seed)
# Returns a tuple of two arrays, first is grad phi wrt length_scale, second is grad phi wrt sigma_RBF
function gradfeatureNotensor(X::Array,length_scale::Real,sigma_RBF::Real,seed::Integer,phi::Array)
	D=size(X,2);
	n=size(phi,1);	
	srand(seed)
	Z=randn(n,D)
	return phi.*(Z*X'),phi/sigma_RBF
end

# alternative fourier feature embedding for the no tensor model (full-theta) using fixed length_scales
function featureNotensor2(X::Array,n::Integer,length_scale::Real,sigma_RBF::Real,seed::Integer)
	if n%2==0
		half_n=int(n/2)
		N,D=size(X)
		phi=Array(Float64,n,N)
		srand(seed)
		Z=randn(half_n,D)/length_scale
		for i=1:N
		    for j=1:half_n
		        temp=sum(X[i,:].*Z[j,:])
		        phi[2*j-1,i]=sin(temp)
		        phi[2*j,i]=cos(temp)
		    end                    
		end
		return sqrt(2/n)*sigma_RBF*phi
    else error("n is not even")
    end
end

# alternative fourier feature embedding for the no tensor model (full-theta) using varying length_scales
function featureNotensor2(X::Array,n::Integer,length_scale::Vector,sigma_RBF::Real,seed::Integer)
	if n%2==0
		half_n=int(n/2)
		N,D=size(X)
		if length(length_scale)!=D
			error("dimensions of X and length_scale do not match")
		end
		phi=Array(Float64,n,N)
		srand(seed)
		Z=Array(Float64,half_n,D)
		for k=1:D
			Z[:,k]=randn(half_n)/length_scale[k]
		end
		for i=1:N
		    for j=1:half_n
		        temp=sum(X[i,:].*Z[j,:])
		        phi[2*j-1,i]=sin(temp)
		        phi[2*j,i]=cos(temp)
		    end                    
		end
		return sqrt(2/n)*sigma_RBF*phi
    else error("n is not even")
    end
end

# sample the Q random non-zero locations of w
function samplenz(r::Integer,D::Integer,Q::Integer,seed::Integer)
    srand(seed)
    L=sample(0:(r^D-1),Q,replace=false)
    I=Array(Int32,Q,D)
    for q in 1:Q
        I[q,:]=digits(L[q],r,D)+1
    end
    # this way the locations are drawn uniformly from the lattice [r^D] without replacement
    # so I_qd=index of dth dim of qth non-zero
    return I
end

#compute <phi^(k)(x_i),U^(k)_{.l}> for all k,l,batch
function phidotU(U::Array,phi::Array)
    n,D,data_size=size(phi)
    r=size(U,2)
    temp=Array(Float64,D,r,data_size)
    for i=1:data_size
        for l=1:r
            for k=1:D
		temp[k,l,i]=dot(phi[:,k,i],U[:,l,k])
	    end
        end
    end
    return temp
end

#compute V st V[q,i]=prod_{k=1 to D}(temp[k,I[q,k],i])
function computeV(temp::Array,I::Array)
    Q,D=size(I);
    data_size=size(temp,3)
    V=ones(Q,data_size)
    for i=1:data_size
        for q=1:Q
	    for k=1:D
		V[q,i]*=temp[k,I[q,k],i];
	    end
        end
    end
    return V
end

#compute predictions fhat from V,w
function computefhat(V::Array,w::Array)
    data_size=size(V,2)
    fhat=Array(Float64,data_size)
    for i=1:data_size
	fhat[i]=dot(V[:,i],w)
    end
    return fhat
end

#compute predictions from w,U,I
function pred(w::Array,U::Array,I::Array,phitest::Array)

    # compute <phi^(k)(x_i),U^(k)_{.l}> for all k,l,test and store in temp
    temp=phidotU(U,phitest)

    # compute V st V[q,i]=prod_{k=1 to D}(temp[k,I[q,k],i])
    V=computeV(temp,I)

    # compute fhat where fhat[i]=V[:,i]'w
    return computefhat(V,w)
end

# compute U_phi[q,i,k]=expression in big brackets in (11)
function computeU_phi(V::Array,temp::Array,I::Array)
    Q,D=size(I)
    data_size=size(V,2)
    U_phi=Array(Float64,Q,data_size,D)
    for k=1:D
        for i=1:data_size
            for q=1:Q
	        U_phi[q,i,k]=V[q,i]/temp[k,I[q,k],i]
            end
	end
    end
    return U_phi
end

# compute a_l^(k)(x_i) for l=1,...,r k=1,..,D and store in A
function computeA(U_phi::Array,w::Array,I::Array,r::Integer)
    Q,data_size,D=size(U_phi)
    A=zeros(r,D,data_size)
    for i=1:data_size
        for k=1:D
            for l in unique(I[:,k])
                index=findin(I[:,k],l) #I_l
                A[l,k,i]=dot(U_phi[index,i,k],w[index]) 
            end
        end
    end
    return A
end

# compute Psi as in (12)
function computePsi(A::Array,phi::Array)
    r,D,data_size=size(A)
    n,D,data_size=size(phi)
    Psi=Array(Float64,n*r,data_size,D)
    for i=1:data_size
        for k=1:D
            Psi[:,i,k]=kron(A[:,k,i],phi[:,k,i])
        end
    end
    return Psi
end

#work out minimum RMSE by averaging over predictions in w_store,U_store, returning both RMSE and meanhat
function RMSE(w_store::Array,U_store::Array,I::Array,phitest::Array,ytest::Array)
    Ntest=length(ytest);
    T=size(w_store,2);
    meanfhat=zeros(Ntest);
    for i=1:T
        meanfhat+=pred(w_store[:,i],U_store[:,:,:,i],I,phitest);
    end
    meanfhat=meanfhat/T;
    return norm(ytest-meanfhat)/sqrt(Ntest),meanfhat;
end

#work out minimum RMSE by averaging over predictions in w_store,U_store, returning both RMSE and meanhat
#parallel version
function parallelRMSE(w_store::Array,U_store::Array,I::Array,phitest::Array,ytest::Array)
    Ntest=length(ytest);
    T=size(w_store,2);
    meanfhat= @parallel (+) for i=1:T
        pred(w_store[:,i],U_store[:,:,:,i],I,phitest);
    end
    meanfhat=meanfhat/T;
    return norm(ytest-meanfhat)/sqrt(Ntest),meanfhat;
end

#create mesh for GPT_demo
function createmesh(interval_start,interval_end,npts)
    x=linspace(interval_start,interval_end,npts)
    y=linspace(interval_start,interval_end,npts)
    grid=Array(Float64,npts^2,2); k=1;
    for i=1:npts
        for j=1:npts
            grid[k,:]=[x[i] y[j]];
            k+=1;
        end
    end
    #grid=[x[1] y[1]; x[1] y[2]; ...]
    return x,y,grid
end

#draw fhat from Tensor model for GPT_demo for same length_scale across dimensions
function fhatdraw(X::Array,n::Integer,length_scale::Real,sigma_RBF::Real,r::Integer,Q::Integer)
    N,D=size(X)
    scale=sqrt(n/(Q^(1/D)));
    seed=17;
    phi=feature(X,n,length_scale,sigma_RBF,seed,scale)
    sigma_w=1;
    w=sigma_w*randn(Q)
    U=Array(Float64,n,r,D)
    for k=1:D
        Z=randn(r,n)
        U[:,:,k]=transpose(\(sqrtm(Z*Z'),Z)) #sample uniformly from V_{n,r}
    end
    I=samplenz(r,D,Q,seed)
    
    return pred(w,U,I,phi)
end

#draw fhat from Tensor model for GPT_demo for varying length_scale
function fhatdraw(X::Array,n::Integer,length_scale::Vector,sigma_RBF::Real,r::Integer,Q::Integer)
    N,D=size(X)
	if length(length_scale)!=D
			error("dimensions of X and length_scale do not match")
	end
    scale=sqrt(n/(Q^(1/D)));
    seed=17;
    phi=feature(X,n,length_scale,sigma_RBF,seed,scale)
    sigma_w=1;
    w=sigma_w*randn(Q)
    U=Array(Float64,n,r,D)
    for k=1:D
        Z=randn(r,n)
        U[:,:,k]=transpose(\(sqrtm(Z*Z'),Z)) #sample uniformly from V_{n,r}
    end
    I=samplenz(r,D,Q,seed)
    
    return pred(w,U,I,phi)
end

    
#SGLD on Tucker Model with Stiefel Manifold
function GPT_SGLDERM(phi::Array, y::Array, signal_var::Real, I::Array, r::Integer, Q::Integer, m::Integer, epsw::Real, epsU::Real, burnin::Integer, maxepoch::Integer)
    # phi is the D by n by N array of features where phi[k,:,i]=phi^(k)(x_i)
    # signal_var is the variance of the observed values
    # epsw,epsU are the epsilons for w and U resp.
    # maxepoch is the number of sweeps through whole dataset
    
    n,D,N=size(phi)
    numbatches=int(ceil(N/m))
    sigma_w=1;
    
    # initialise w,U^(k)
    w_store=Array(Float64,Q,maxepoch*numbatches)
    U_store=Array(Float64,n,r,D,maxepoch*numbatches)
    w=sigma_w*randn(Q)

    U=Array(Float64,n,r,D)
    for k=1:D
        Z=randn(r,n)
        U[:,:,k]=transpose(\(sqrtm(Z*Z'),Z)) #sample uniformly from V_{n,r}
    end


    for epoch=1:(burnin+maxepoch)
        #randomly permute training data and divide into mini_batches of size m
        perm=randperm(N)
        phi=phi[:,:,perm]; y=y[perm];
        
        # run SGLD on w and SGLDERM on U
        for batch=1:numbatches
            # random samples for the stochastic gradient
            idx=(m*(batch-1)+1):min(m*batch,N)
            phi_batch=phi[:,:,idx]; y_batch=y[idx];
            batch_size=length(idx) #this is m except for last batch

            # compute <phi^(k)(x_i),U^(k)_{.l}> for all k,l,batch and store in temp
            temp=phidotU(U,phi_batch)

	    # compute V st V[q,i]=prod_{k=1 to D}(temp[k,I[q,k],i])
            V=computeV(temp,I)
	    
            # compute fhat where fhat[i]=V[:,i]'w
            fhat=computefhat(V,w)

            # now can compute gradw, the stochastic gradient of log post wrt w
            gradw=(N/batch_size)*V*(y_batch-fhat)/signal_var-w/(sigma_w^2)

            # compute U_phi[q,i,k]=expression in big brackets in (11)
            U_phi=computeU_phi(V,temp,I)
            
            # compute a_l^(k)(x_i) for l=1,...,r k=1,..,D and store in A
            A=computeA(U_phi,w,I,r)
            
            # compute Psi as in (12)
            Psi=computePsi(A,phi_batch)
            
            # can now compute gradU where gradU[:,:,k]=stochastic gradient of log post wrt U^(k)
            gradU=Array(Float64,n,r,D)
            for k=1:D
                gradU[:,:,k]=reshape((N/batch_size)*Psi[:,:,k]*(y_batch-fhat)/signal_var,n,r)
            end
	    
            # SGLD step on w
            w[:]+=epsw*gradw/2 +sqrt(epsw)*randn(Q)
	    #if batch==1
	    #	println("mean epsgradw_half=",mean(epsw*gradw/2)," std =",std(epsw*gradw/2))
	    #	println("meansqrtepsgradU_half=",mean(sqrt(epsU)*gradU/2), " std=",std(sqrt(epsU)*gradU/2))
	    #end
            # SGLDERM step on U
            for k=1:D
                mom=proj(U[:,:,k],sqrt(epsU)*gradU[:,:,k]/2+randn(n,r))
                U[:,:,k]=geod(U[:,:,k],mom,sqrt(epsU));
                if U[:,:,k]==zeros(n,r) #if NaN appears while evaluating G
                    return zeros(Q,maxepoch*numbatches),zeros(n,r,D,maxepoch*numbatches)
                end
            end
	    if epoch>burnin
	        w_store[:,((epoch-burnin)-1)*numbatches+batch]=w
	        U_store[:,:,:,((epoch-burnin)-1)*numbatches+batch]=U
	    end
        end
    end
    return w_store,U_store
end

#SGD on Tucker Model with Stiefel Manifold 
function GPT_SGDERM(phi::Array, y::Array, signal_var::Real, I::Array, r::Integer, Q::Integer, m::Integer, epsw::Real, epsU::Real, burnin::Integer, maxepoch::Integer)
    # phi is the D by n by N array of features where phi[k,:,i]=phi^(k)(x_i)
    # signal_var is the variance of the observed values
    # epsw,epsU are the epsilons for w and U resp.
    # maxepoch is the number of sweeps through whole dataset
    
    n,D,N=size(phi)
    numbatches=int(ceil(N/m))
    sigma_w=1;
    
    # initialise w,U^(k)
    w_store=Array(Float64,Q,maxepoch*numbatches)
    U_store=Array(Float64,n,r,D,maxepoch*numbatches)
    w=sigma_w*randn(Q)

    U=Array(Float64,n,r,D)
    for k=1:D
        Z=randn(r,n)
        U[:,:,k]=transpose(\(sqrtm(Z*Z'),Z)) #sample uniformly from V_{n,r}
    end

    for epoch=1:(burnin+maxepoch)
        #randomly permute training data and divide into mini_batches of size m
        perm=randperm(N)
        phi=phi[:,:,perm]; y=y[perm];
        
        # run SGLD on w and SGLDERM on U
        for batch=1:numbatches
            # random samples for the stochastic gradient
            idx=(m*(batch-1)+1):min(m*batch,N)
            phi_batch=phi[:,:,idx]; y_batch=y[idx];
            batch_size=length(idx) #this is m except for last batch

            # compute <phi^(k)(x_i),U^(k)_{.l}> for all k,l,batch and store in temp
            temp=phidotU(U,phi_batch)

	    # compute V st V[q,i]=prod_{k=1 to D}(temp[k,I[q,k],i])
            V=computeV(temp,I)
	    
            # compute fhat where fhat[i]=V[:,i]'w
            fhat=computefhat(V,w)

            # now can compute gradw, the stochastic gradient of log post wrt w
            gradw=(N/batch_size)*V*(y_batch-fhat)/signal_var-w/(sigma_w^2)

            # compute U_phi[q,i,k]=expression in big brackets in (11)
            U_phi=computeU_phi(V,temp,I)
            
            # compute a_l^(k)(x_i) for l=1,...,r k=1,..,D and store in A
            A=computeA(U_phi,w,I,r)
            
            # compute Psi as in (12)
            Psi=computePsi(A,phi_batch)
            
            # can now compute gradU where gradU[:,:,k]=stochastic gradient of log post wrt U^(k)
            gradU=Array(Float64,n,r,D)
            for k=1:D
                gradU[:,:,k]=reshape((N/batch_size)*Psi[:,:,k]*(y_batch-fhat)/signal_var,n,r)
            end
	    
            # SGLD step on w
            w[:]+=epsw*gradw/2
	    #if batch==1
	    #	println("mean epsgradw_half=",mean(epsw*gradw/2)," std =",std(epsw*gradw/2))
	    #	println("meansqrtepsgradU_half=",mean(sqrt(epsU)*gradU/2), " std=",std(sqrt(epsU)*gradU/2))
	    #end
            # SGLDERM step on U
            for k=1:D
                mom=proj(U[:,:,k],sqrt(epsU)*gradU[:,:,k]/2)
                U[:,:,k]=geod(U[:,:,k],mom,sqrt(epsU));
                if U[:,:,k]==zeros(n,r) #if NaN appears while evaluating G
                    return zeros(Q,maxepoch*numbatches),zeros(n,r,D,maxepoch*numbatches)
                end
            end
	    if epoch>burnin
	        w_store[:,((epoch-burnin)-1)*numbatches+batch]=w
	        U_store[:,:,:,((epoch-burnin)-1)*numbatches+batch]=U
	    end
        end
    end
    return w_store,U_store
end

#GMC on Tucker Model with Stiefel Manifold
function GPT_GMC(phi::Array, y::Array, signal_var::Real, I::Array, r::Integer, Q::Integer, epsw::Real, epsU::Real, burnin::Integer, maxepoch::Integer, L::Integer,param_seed::Integer)
    # phi is the D by n by N array of features where phi[k,:,i]=phi^(k)(x_i)
    # signal_var is the variance of the observed values
    # epsw,epsU are the epsilons for w and U resp.
    # maxepoch is the number of sweeps through whole dataset
    
    n,D,N=size(phi)
    sigma_w=1;
    
    # initialise w,U^(k)
    w_store=Array(Float64,Q,maxepoch)
    U_store=Array(Float64,n,r,D,maxepoch)
    accept_prob=Array(Float64,maxepoch+burnin)
    srand(param_seed)
    w=sigma_w*randn(Q)

    U=Array(Float64,n,r,D)
    for k=1:D
        Z=randn(r,n)
        U[:,:,k]=transpose(\(sqrtm(Z*Z'),Z)) #sample uniformly from V_{n,r}
    end

    for epoch=1:(burnin+maxepoch)
        w_old=w; U_old=U;
        # initialise momentum terms and Hamiltonian
        p=randn(Q); mom=Array(Float64,n,r,D);
        for k=1:D
            mom[:,:,k]=proj(U[:,:,k],randn(n,r));
        end
        H_old=-sum(w.*w)/(2*sigma_w^2)-norm(y-pred(w,U,I,phi))^2/(2*signal_var)-sum(mom.*mom)/2-sum(p.*p)/2;
        #println("H=",H_old)
        pred_new=Array(Float64,N); # used later for computing H_new
        # leapfrog step
        for l=1:L
            ### update p and mom
            # compute <phi^(k)(x_i),U^(k)_{.l}> for all k,l,i and store in temp
            temp=phidotU(U,phi)
	    # compute V st V[q,i]=prod_{k=1 to D}(temp[k,I[q,k],i])
            V=computeV(temp,I)
            # compute fhat where fhat[i]=V[:,i]'w
            fhat=computefhat(V,w)
            # now can compute gradw, the stochastic gradient of log post wrt w
            gradw=V*(y-fhat)/(signal_var)-w/(sigma_w^2)
            # update p
            p+=sqrt(epsw)*gradw/2;
            # compute U_phi[q,i,k]=expression in big brackets in (11)
            U_phi=computeU_phi(V,temp,I)
            # compute a_l^(k)(x_i) for l=1,...,r k=1,..,D and store in A
            A=computeA(U_phi,w,I,r)
            # compute Psi as in (12)
            Psi=computePsi(A,phi)
            # can now compute gradU where gradU[:,:,k]=gradient of log post wrt U^(k)
            gradU=Array(Float64,n,r,D)
            for k=1:D
                gradU[:,:,k]=reshape(Psi[:,:,k]*(y-fhat)/(signal_var),n,r)
            end
            # update mom
            for k=1:D
                mom[:,:,k]=proj(U[:,:,k],mom[:,:,k]+sqrt(epsU)*gradU[:,:,k]/2)
            end

            ### update w,U,mom 
            # update w
            w+=sqrt(epsw)*p
            # update U and mom
            for k=1:D
                U[:,:,k],mom[:,:,k]=geodboth(U[:,:,k],mom[:,:,k],sqrt(epsU));
                if U[:,:,k]==zeros(n,r) #if NaN appears while evaluating G
                    return zeros(Q,maxepoch),zeros(n,r,D,maxepoch),zeros(maxepoch+burnin)*NaN
                end
            end

            ### update p and mom with new w,U
            # compute <phi^(k)(x_i),U^(k)_{.l}> for all k,l,i and store in temp
            temp=phidotU(U,phi)
	    # compute V st V[q,i]=prod_{k=1 to D}(temp[k,I[q,k],i])
            V=computeV(temp,I)
            # compute fhat where fhat[i]=V[:,i]'w
            fhat=computefhat(V,w)
            # now can compute gradw, the stochastic gradient of log post wrt w
            gradw=V*(y-fhat)/(signal_var)-w/(sigma_w^2)
            # update p
            p+=sqrt(epsw)*gradw/2;
            # compute U_phi[q,i,k]=expression in big brackets in (11)
            U_phi=computeU_phi(V,temp,I)
            # compute a_l^(k)(x_i) for l=1,...,r k=1,..,D and store in A
            A=computeA(U_phi,w,I,r)
            # compute Psi as in (12)
            Psi=computePsi(A,phi)
            # can now compute gradU where gradU[:,:,k]=gradient of log post wrt U^(k)
            gradU=Array(Float64,n,r,D)
            for k=1:D
                gradU[:,:,k]=reshape(Psi[:,:,k]*(y-fhat)/signal_var,n,r)
            end
            # update mom
            for k=1:D
                mom[:,:,k]=proj(U[:,:,k],mom[:,:,k]+sqrt(epsU)*gradU[:,:,k]/2)
            end

            # store the last prediction for computing H_new
            if l==L
                pred_new=fhat;
            end
        end

        H=-sum(w.*w)/(2*sigma_w^2)-norm(y-pred_new)^2/(2*signal_var)-sum(mom.*mom)/2-sum(p.*p)/2;
        u=rand(1);
        
        accept_prob[epoch]=exp(H-H_old)
        println("accept_prob=",accept_prob[epoch])
        
        if u[1]>accept_prob[epoch] #if true, reject 
            w=w_old; U=U_old;
        end
        
	if epoch>burnin
	    w_store[:,epoch-burnin]=w
	    U_store[:,:,:,epoch-burnin]=U
        end
    end
    return w_store,U_store,accept_prob
end
    

#SGLD on No Tensor Model
function GPNT_SGLD(phi::Array, y::Array, signal_var::Real, sigma_theta::Real, m::Integer, eps_theta::Real, decay_rate::Real, burnin::Integer, maxepoch::Integer, theta_seed::Integer)
    n,N=size(phi);
    numbatches=int(ceil(N/m))
    
    #initialise theta
    srand(theta_seed)
    theta=sigma_theta*randn(n);
    epsilon=eps_theta;
    theta_store=Array(Float64,n,(maxepoch+burnin)*numbatches)
    meangrad=zeros(n);

    t=0; #iteration number
    g=0;
    for epoch=1:(maxepoch+burnin)
        #randomly permute training data and divide into mini_batches of size m
        perm=randperm(N)
        phi=phi[:,perm]; y=y[perm];
        
        # run SGLD on w and SGLDERM on U
        for batch=1:numbatches
	    t+=1;
            # random samples for the stochastic gradient
            idx=(m*(batch-1)+1):min(m*batch,N)
            phi_batch=phi[:,idx]; y_batch=y[idx];
            batch_size=length(idx) #this is m except for last batch

            epsilon=eps_theta*t^(-decay_rate)
            grad_theta=-theta/(sigma_theta^2)+(N/batch_size)*phi_batch*(y_batch-phi_batch'*theta)/signal_var;
            grad=epsilon*grad_theta/2;
            noise=sqrt(epsilon)*randn(n);
           
            theta[:]+=grad+noise;
	    theta_store[:,t]=theta;
	    if epoch>burnin && batch<numbatches
		g+=1;
            	meangrad+=phi_batch*(y_batch-phi_batch'*theta)/signal_var;
	    end
        end
    end
    meangrad=meangrad/g;
    return theta_store,meangrad
end

# function to return the negative log marginal likelihood of No Tensor model with Gaussian likelihood and fixed length_scale
function GPNT_logmarginal(X::Array,y::Array,n::Integer,length_scale::Real,sigma_RBF::Real,signal_var::Real,seed::Integer)
    N,D=size(X);
    phi=featureNotensor(X,n,length_scale,sigma_RBF,seed);
    A=phi*phi'+signal_var*eye(n);
    b=phi*y;
	B=\(A,b);
	lambda=eigvals(A);
	logdetA=sum(log(lambda));
    return (N-n)*log(signal_var)/2+logdetA/2+(sum(y.*y)-sum(b.*B))/(2*signal_var)
end

# function to return the negative log marginal likelihood of No Tensor model with Gaussian likelihood and varying length_scale
function GPNT_logmarginal(X::Array,y::Array,n::Integer,length_scale::Vector,sigma_RBF::Real,signal_var::Real,seed::Integer)
    N,D=size(X);
	if length(length_scale)!=D
			error("dimensions of X and length_scale do not match")
	end
    phi=featureNotensor(X,n,length_scale,sigma_RBF,seed);
    A=phi*phi'+signal_var*eye(n);
    b=phi*y;
	B=\(A,b);
	lambda=eigvals(A);
	logdetA=sum(log(lambda));
    return (N-n)*log(signal_var)/2+logdetA/2+(sum(y.*y)-sum(b.*B))/(2*signal_var)
end

# learning hyperparams signal_var,sigma_RBF,length_scale for No Tensor Model by optimising Gaussian marginal likelihood for fixed length_scale
function GPNT_hyperparameters(X::Array,y::Array,n::Integer,init_length_scale::Real,init_sigma_RBF::Real,init_signal_var::Real,seed::Integer)
	D=size(X,2);
    logmarginal(hyperparams::Vector)=GPNT_logmarginal(X,y,n,exp(hyperparams[1]),exp(hyperparams[2]),exp(hyperparams[3]),seed); # log marginal likelihood as a fn of hyperparams=log([length_scale,sigma_RBF,signal_var]) only.
    # exp needed to enable unconstrained optimisation, since length_scale,sigmaRBF,signal_var must be positive
    g=ForwardDiff.gradient(logmarginal)
    function g!(hyperparams::Vector,storage::Vector)
        grad=g(hyperparams)
        for i=1:length(hyperparams)
            storage[i]=grad[i]
        end
    end
    l=optimize(logmarginal,g!,log([init_length_scale,init_sigma_RBF,init_signal_var]),method=:cg,show_trace = true, extended_trace = true)
	return exp(l.minimum)
end

#learning hyperparams signal_var,sigma_RBF,length_scale for No Tensor Model by optimising Gaussian marginal likelihood for varying length_scale
function GPNT_hyperparameters(X::Array,y::Array,n::Integer,init_length_scale::Vector,init_sigma_RBF::Real,init_signal_var::Real,seed::Integer)
	D=size(X,2);
    logmarginal(hyperparams::Vector)=GPNT_logmarginal(X,y,n,exp(hyperparams[1:D]),exp(hyperparams[D+1]),exp(hyperparams[D+2]),seed); # log marginal likelihood as a fn of hyperparams=log([length_scale,sigma_RBF,signal_var]) only.
    # exp needed to enable unconstrained optimisation, since length_scale,sigmaRBF,signal_var must be positive
    g=ForwardDiff.gradient(logmarginal)
    function g!(hyperparams::Vector,storage::Vector)
        grad=g(hyperparams)
        for i=1:length(hyperparams)
            storage[i]=grad[i]
        end
    end
    l=optimize(logmarginal,g!,log([init_length_scale,init_sigma_RBF,init_signal_var]),method=:cg,show_trace = true, extended_trace = true)
	return exp(l.minimum)
end

# function to learn hyperparams signal_var,sigma_RBF,length_scale for No Tensor Model by optimising non-Gaussian marginal likelihood using the stochastic EM algorithm for fixed length_scale
function GPNT_hyperparameters_ng(init_theta::Vector,init_length_scale,init_sigma_RBF::Real,init_signal_var::Real,
neglogjointlkhd::Function,gradneglogjointlkhd::Function,epsilon::Real=1e-5,num_cg_iter::Integer=10,num_sgld_iter::Integer=10)
	# neglogjointlkhd should be -log p(y,theta;hyperparams), a function with 
	# input theta,length_scale,sigma_RBF,signal_var and scalar output

	# gradneglogjointlkhd should be the gradient of neglogjointlkhd wrt theta and hyperparams with
	# input theta,length_scale,sigma_RBF,signal_var and vector output of length equal to length(theta)+3

	# initialise theta and loghyperparams
	theta=init_theta; loghyperparams=[log(init_length_scale),log(init_sigma_RBF),log(init_signal_var)];
	n=length(init_theta);

	# define f which corresponds to neglogjointlkhd and gradneglogjointlkhd but with inputs loghyperparams instead of hyperparams (for optimisation's sake) and g its gradient - compute using chain rule
	f(theta,loglength_scale,logsigma_RBF,logsignal_var)=neglogjointlkhd(theta,exp(loglength_scale),exp(logsigma_RBF),exp(logsignal_var));
	g(theta,loglength_scale,logsigma_RBF,logsignal_var)=gradneglogjointlkhd(theta,exp(loglength_scale),exp(logsigma_RBF),exp(logsignal_var)).*[ones(n),exp(loglength_scale),exp(logsigma_RBF),exp(logsignal_var)];

	# stochastic EM 

	# initialise statistic for diagnosing convergence
	absdiff=1; # |x - x'|
	iter=1;
	while absdiff>1e-7
		println("iteration ",iter)
		# E step - sample theta from posterior using SGLD - but then need to decide on step size
		for i=1:num_sgld_iter
			theta-=epsilon*g(theta,loghyperparams...)[1:n]+sqrt(epsilon)*randn(n)
		end
		println("theta norm=",norm(theta))
		# M step - maximisie joint log likelihood wrt hyperparams using no_cg_iter steps of cg/gd

		f2(loghyperparameters::Vector)=f(theta,loghyperparameters...);
		g2(loghyperparameters::Vector)=g(theta,loghyperparameters...)[end-length(loghyperparameters)+1:end];
		function g2!(loghyperparameters::Vector,storage::Vector)
			grad=g2(loghyperparameters)
			for i=1:length(loghyperparameters)
		    	storage[i]=grad[i]
			end
		end
		l=optimize(f2,g2!,loghyperparams,method=:cg,show_trace = true, extended_trace = true, iterations=num_cg_iter)
		new_loghyperparams=l.minimum

		#update convergence statistics
		absdiff=norm(exp(loghyperparams)-exp(new_loghyperparams));
		println("|x-x'|: ",absdiff)
		println()

		#update hyperparams
		loghyperparams=new_loghyperparams
		iter+=1
	end
	
	return exp(loghyperparams)
#=
	function g!(params::Vector,storage::Vector)
		grad=g(params)
		for i=1:length(params)
		    storage[i]=grad[i]
		end
	end

	# optimise posterior of theta wrt theta and hyperparams - simplification of stochastic EM below
	l=optimize(f,g!,[init_theta,loghyperparams],method=:cg,show_trace = true, extended_trace = true)
	return exp(l.minimum[end-2:end])
=#	
end	

function GPT_SGLDERMw(phi::Array, y::Array, signal_var::Real, I::Array, r::Integer, Q::Integer, m::Integer, epsw::Real, burnin::Integer, maxepoch::Integer)
    # phi is the D by n by N array of features where phi[k,:,i]=phi^(k)(x_i)
    # signal_var is the variance of the observed values
    # epsw,epsU are the epsilons for w and U resp.
    # maxepoch is the number of sweeps through whole dataset
    
    n,D,N=size(phi)
    numbatches=int(ceil(N/m))
    sigma_w=1;
    
    # initialise w,U^(k)
    w_store=Array(Float64,Q,maxepoch*numbatches)
    w=sigma_w*randn(Q)

    U=Array(Float64,n,r,D)
    for k=1:D
        Z=randn(r,n)
        U[:,:,k]=transpose(\(sqrtm(Z*Z'),Z)) #sample uniformly from V_{n,r}
    end


    for epoch=1:(burnin+maxepoch)
        #randomly permute training data and divide into mini_batches of size m
        perm=randperm(N)
        phi=phi[:,:,perm]; y=y[perm];
        
        # run SGLD on w and SGLDERM on U
        for batch=1:numbatches
            # random samples for the stochastic gradient
            idx=(m*(batch-1)+1):min(m*batch,N)
            phi_batch=phi[:,:,idx]; y_batch=y[idx];
            batch_size=length(idx) #this is m except for last batch

            # compute <phi^(k)(x_i),U^(k)_{.l}> for all k,l,batch and store in temp
            temp=phidotU(U,phi_batch)

	    # compute V st V[q,i]=prod_{k=1 to D}(temp[k,I[q,k],i])
            V=computeV(temp,I)
	    
            # compute fhat where fhat[i]=V[:,i]'w
            fhat=computefhat(V,w)

            # now can compute gradw, the stochastic gradient of log post wrt w
            gradw=(N/batch_size)*V*(y_batch-fhat)/signal_var-w/(sigma_w^2)
	    
            # SGLD step on w
            w[:]+=epsw*gradw/2 +sqrt(epsw)*randn(Q)
           
	    if epoch>burnin
	        w_store[:,((epoch-burnin)-1)*numbatches+batch]=w
	    end
        end
    end
    return w_store,U
end

#SGLD with RMSprop on Tucker Model with Stiefel Manifold
function GPT_SGLDERM_RMSprop(phi::Array, y::Array, signal_var::Real, I::Array, r::Integer, Q::Integer, m::Integer, epsilon::Real, alpha::Real, burnin::Integer, maxepoch::Integer)
    # phi is the D by n by N array of features where phi[k,:,i]=phi^(k)(x_i)
    # signal_var is the variance of the observed values
    # epsilon is the stepsize for U and w
    # alpha is the coeff for moving average in RMSprop (usually 0.9 or higher)
    # maxepoch is the number of sweeps through whole dataset
    
    n,D,N=size(phi)
    numbatches=int(ceil(N/m))
    sigma_w=1;
    
    # initialise w,U and gw,gU (moving average of gradients for RMSprop)
    w_store=Array(Float64,Q,maxepoch*numbatches)
    U_store=Array(Float64,n,r,D,maxepoch*numbatches)
    w=sigma_w*randn(Q)

    U=Array(Float64,n,r,D)
    for k=1:D
        Z=randn(r,n)
        U[:,:,k]=transpose(\(sqrtm(Z*Z'),Z)) #sample uniformly from V_{n,r}
    end

    gw=zeros(Q)
    gU=zeros(n,r,D)

    lambda=1e-5 #smoothing value for numerical convenience in RMSprop

    for epoch=1:(burnin+maxepoch)
        #randomly permute training data and divide into mini_batches of size m
        perm=randperm(N)
        phi=phi[:,:,perm]; y=y[perm];
        
        # run SGLD on w and SGLDERM on U
        for batch=1:numbatches
            # random samples for the stochastic gradient
            idx=(m*(batch-1)+1):min(m*batch,N)
            phi_batch=phi[:,:,idx]; y_batch=y[idx];
            batch_size=length(idx) #this is m except for last batch

            # compute <phi^(k)(x_i),U^(k)_{.l}> for all k,l,batch and store in temp
            temp=phidotU(U,phi_batch)

	    # compute V st V[q,i]=prod_{k=1 to D}(temp[k,I[q,k],i])
            V=computeV(temp,I)
	    
            # compute fhat where fhat[i]=V[:,i]'w
            fhat=computefhat(V,w)

            # now can compute gradw, the unnormalised stochastic gradient of log lik wrt w
            gradw=(1/batch_size)*V*(y_batch-fhat)/signal_var

            # update gw and compute step size for w
            gw=alpha*gw+(1-alpha)*(gradw.^2) 
            epsw=epsilon./(sqrt(gw)+lambda)
			
			#=
			if batch==numbatches
				println("epoch=",epoch)            
				println("norm of gw=",sqrt(sum(gw.^2)),";std of gw=",std(gw))
            	println("norm of epsw=",sqrt(sum(epsw.^2)),";std of epsw=",std(epsw))
            end
			=#

            # normalise stochastic grad and add grad of log prior to make grad of log post
            gradw=N*gradw-w/(sigma_w^2)

            # SGLD step on w with step size given by RMSprop
            w[:]+=epsw.*gradw/2 +sqrt(epsw).*randn(Q)

            # compute U_phi[q,i,k]=expression in big brackets in (11)
            U_phi=computeU_phi(V,temp,I)
            
            # compute a_l^(k)(x_i) for l=1,...,r k=1,..,D and store in A
            A=computeA(U_phi,w,I,r)
            
            # compute Psi as in (12)
            Psi=computePsi(A,phi_batch)
            
            # can now compute gradU where gradU[:,:,k]=unnormalised stochastic grad of log post wrt U^(k)
            gradU=Array(Float64,n,r,D)
            for k=1:D
                gradU[:,:,k]=reshape((1/batch_size)*Psi[:,:,k]*(y_batch-fhat)/signal_var,n,r)
            end

            # update gU and compute step size for U
            # Note we can only have one step size for each U^(k) since we must move along geodesics for a scalar time
            # Hence epsU  needs to be averaged for each U^(k)
            gU=alpha*gU+(1-alpha)*(gradU.^2)
            epsU=epsilon./(sqrt(gU)+lambda)
            meanepsU=vec(mean(epsU,[1,2]));

			#=
			if batch==numbatches
            	println("norm of gU=",sqrt(sum(gU.^2)),";std of gU=",std(gU))
            	println("norm of meanepsU=",sqrt(sum(meanepsU.^2)),";std of meanepsU=",std(meanepsU))
			end
			=#

            # normalise stochastic grad (note log prior of U is const since it is uniform)
            gradU*=N
            
            # SGLDERM step on U with step size given by RMSprop
            for k=1:D
                mom=proj(U[:,:,k],sqrt(meanepsU[k])*gradU[:,:,k]/2+randn(n,r))
                U[:,:,k]=geod(U[:,:,k],mom,sqrt(meanepsU[k]));
                if U[:,:,k]==zeros(n,r) #if NaN appears while evaluating G
                    return zeros(Q,maxepoch*numbatches),zeros(n,r,D,maxepoch*numbatches)
                end
            end
			if epoch>burnin
			    w_store[:,((epoch-burnin)-1)*numbatches+batch]=w
			    U_store[:,:,:,((epoch-burnin)-1)*numbatches+batch]=U
			end
        end
    end
    return w_store,U_store
end


end
    
    
