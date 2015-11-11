module GPT_SGLD

using Distributions
using PyPlot

export proj,geod,datawhitening,feature,pred,RMSE,GPT_SGLDERM,SDexp
    
# define proj for Stiefel manifold
function proj(U::Array,V::Array)
    return V-U*(U'*V+V'*U)/2
end

# define geod for Stiefel manifold
function geod(U::Array,mom::Array,t::Real)
    n,r=size(U)
    A=U'*mom
    temp=[A -mom'*mom;eye(r) A]
    E=expm(t*temp) #can become NaN when temp too large. Return 0 in this case
    if sum(isnan(E))>0
        return zeros(n,r)
    else
        tmp=[U mom]*E[:,1:r]*expm(-t*A)
        #ensure that tmp has cols of unit norm
        normconst=Array(Float64,1,r);
        for l=1:r
    	    normconst[1,l]=norm(tmp[:,l])
        end
        return tmp./repmat(normconst,n,1)
    end
end

# centre and normalise data X so that each col has sd=1
function datawhitening(X::Array) 
    for i = 1:size(X,2)   
        X[:,i] = (X[:,i] - mean(X[:,i]))/std(X[:,i])   
    end
    return X
end

#extract features from tensor decomp of each row of X
function feature(X::Array,n::Integer,length_scale::Real,seed::Integer)    
    N,D=size(X)
    phi=Array(Float64,n,D,N)
    srand(seed)
    Z=randn(n,D)/length_scale
    b=randn(n,D)
    for i=1:N
	for k=1:D
	    for j=1:n
		phi[j,k,i]=cos(X[i,k]*Z[j,k]+b[j,k])
	    end
	end
    end
    return sqrt(2/n)*phi
end

#compute predictions from w,U,I
function pred(w::Array,U::Array,I::Array,phitest::Array)
    n,D,test_size=size(phitest)
    Q=length(w)
    r=size(U,2)

    # compute <phi^(k)(x_i),U^(k)_{.l}> for all k,l,batch and store in temp
    temp=Array(Float64,D,r,test_size)
    for i=1:test_size
        for l=1:r
            for k=1:D
		temp[k,l,i]=dot(phitest[:,k,i],U[:,l,k])
	    end
        end
    end

    # compute V st V[q,i]=prod_{k=1 to D}(temp[k,I[q,k],i])
    V=ones(Q,test_size)
    for i=1:test_size
        for q=1:Q
	    for k=1:D
		V[q,i]*=temp[k,I[q,k],i];
	    end
        end
    end

    # compute fhat where fhat[i]=V[:,i]'w
    fhat=Array(Float64,test_size)
    for i=1:test_size
	fhat[i]=dot(V[:,i],w)
    end
    return fhat
end

#work out minimum RMSE by averaging over predictions, starting from last prediction
function RMSE(w_store::Array,U_store::Array,I::Array,phitest::Array,ytest::Array)
    Ntest=length(ytest);
    T=size(w_store,2);
    vecRMSE=Array(Float64,T);
    meanfhat=zeros(Ntest);
    for i=1:T
        meanfhat+=pred(w_store[:,i],U_store[:,:,:,i],I,phitest);
    end
    meanfhat=meanfhat/T;
    return norm(ytest-meanfhat)/sqrt(Ntest);
end

#write RMSE to filename
function SDexp(phitrain,phitest,ytrain,ytest,ytrainStd,seed,sigma,length_scale,n,r,Q,m,epsw,epsU,burnin,maxepoch,filename)
	D=size(phitrain,2);sigmaw=sqrt(n^D/Q); 
	w_store,U_store,I=GPT_SGLDERM(phitrain,ytrain,sigma,sigmaw,r,Q,m,epsw,epsU,burnin,maxepoch);
        predRMSE=RMSE(w_store,U_store,I,phitest,ytest);
	outfile=open(filename,"a") #append to file
	println(outfile,"RMSE=",ytrainStd*predRMSE,";seed=",seed,";sigma=",sigma,";length_scale=",length_scale,";n=",n,";r=",r,";Q=",Q,";m=",m,";epsw=", epsw,";epsU=",epsU,";burnin=",burnin,";maxepoch=",maxepoch);
	close(outfile)
	return w_store,U_store,I
end
    

function GPT_SGLDERM(phi::Array,y::Array,sigma::Real,sigma_w::Real,r::Integer,Q::Integer,m::Integer,epsw::Real,epsU::Real,burnin::Integer,maxepoch::Integer)
    # phi is the D by n by N array of features where phi[k,:,i]=phi^(k)(x_i)
    # sigma is the s.d. of the observed values
    # sigma_w is the s.d. for the Guassian prior on w
    # epsw,epsU are the epsilons for w and U resp.
    # maxepoch is the number of sweeps through whole dataset
    
    n,D,N=size(phi)
    numbatches=int(ceil(N/m))
    
    # initialise w,U^(k)
    w_store=Array(Float64,Q,maxepoch*numbatches)
    U_store=Array(Float64,n,r,D,maxepoch*numbatches)
    w=sigma_w*randn(Q)
    #println("w= ",w)
    U=Array(Float64,n,r,D)
    for k=1:D
        Z=randn(r,n)
        U[:,:,k]=transpose(\(sqrtm(Z*Z'),Z)) #sample uniformly from a*V_{n,r}
    end
    
    # fix the random non-zero locations of w
    L=sample(0:(r^D-1),Q,replace=false)
    I=Array(Int32,Q,D)
    for q in 1:Q
        I[q,:]=digits(L[q],r,D)+1
    end
    # this way the locations are drawn uniformly from the lattice [r^D] without replacement
    # so I_qd=index of dth dim of qth non-zero
    
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
            temp=Array(Float64,D,r,batch_size)
            for i=1:batch_size
                for l=1:r
                    for k=1:D
			temp[k,l,i]=dot(phi_batch[:,k,i],U[:,l,k])
	            end
                end
	    end

	    # compute V st V[q,i]=prod_{k=1 to D}(temp[k,I[q,k],i])
            V=ones(Q,batch_size)
	    for i=1:batch_size
                for q=1:Q
		    for k=1:D
			V[q,i]*=temp[k,I[q,k],i];
		    end
                end
            end
	    
            # compute fhat where fhat[i]=V[:,i]'w
            fhat=Array(Float64,batch_size)
	    for i=1:batch_size
		fhat[i]=dot(V[:,i],w)
	    end
	    #println("epoch=",epoch," batch=",batch," meanfhat= ",mean(fhat)," stdfhat=", std(fhat))
	    #println("epoch=",epoch," batch=",batch," meanw=",mean(w), " stdw=", std(w))

            # now can compute gradw, the stochastic gradient of log post wrt w
            gradw=(N/batch_size)*V*(y_batch-fhat)/(sigma^2)-w/(sigma_w^2)

            # compute U_phi[q,i,k]=expression in big brackets in (11)
            U_phi=Array(Float64,Q,batch_size,D)
            for k=1:D
		for i=1:batch_size
		    for q=1:Q
			U_phi[q,i,k]=V[q,i]/temp[k,I[q,k],i]
		    end
		end
            end
            # now compute a_l^(k)(x_i) for l=1,...,r k=1,..,D and store in A
            A=zeros(r,D,batch_size)
            for i=1:batch_size
                for k=1:D
                    for l in unique(I[:,k])
                        index=findin(I[:,k],l) #I_l
                        A[l,k,i]=dot(U_phi[index,i,k],w[index]) 
                    end
                end
            end
            # compute Psi as in (12)
            Psi=Array(Float64,n*r,batch_size,D)
            for i=1:batch_size
                for k=1:D
                    Psi[:,i,k]=kron(A[:,k,i],phi_batch[:,k,i])
                end
            end
            # can now compute gradU where gradU[:,:,k]=stochastic gradient of log post wrt U^(k)
            gradU=Array(Float64,n,r,D)
            for k=1:D
                gradU[:,:,k]=reshape((N/batch_size)*Psi[:,:,k]*(y[batch]-fhat)/(sigma^2),n,r)
            end

            # SGLD step on w
            w[:]+=epsw*gradw/2 +sqrt(epsw)*randn(Q)
            # SGLDERM step on U
            for k=1:D
                mom=proj(U[:,:,k],sqrt(epsU)*gradU[:,:,k]/2+randn(n,r))
                G=geod(U[:,:,k],mom,sqrt(epsU));
                if G==zeros(n,r) #if NaN appears while evaluating G
                    return zeros(Q,maxepoch*numbatches),zeros(n,r,D,maxepoch*numbatches),I
                else U[:,:,k]=G;
                end
            end
	    if epoch>burnin
	        w_store[:,((epoch-burnin)-1)*numbatches+batch]=w
	        U_store[:,:,:,((epoch-burnin)-1)*numbatches+batch]=U
	    end
        end
    end
    return w_store,U_store,I
end

end
    
    
