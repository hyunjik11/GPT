#implementation with batches
module GPT_SGLD

using Distributions
using PyPlot
using Debug

export proj,geod,datawhitening,feature,pred,RMSE,GPT_SGLDERM
    
# define proj for Stiefel manifold
function proj(U::Array,V::Array)
    return V-U*(U'*V+V'*U)/2
end

# define geod for Stiefel manifold
function geod(U::Array,mom::Array,t::Real)
    n,r=size(U)
    A=U'*mom
    temp=[A -mom'*mom;eye(r) A]
    E=expm(t*temp) #can become NaN when temp too large
    tmp=[U mom]*E[:,1:r]*expm(-t*A)
    #ensure that tmp has cols of norm a
    normconst=Array(Float64,1,r);
    for l=1:r
    	normconst[1,l]=norm(tmp[:,l])
    end
    return tmp./repmat(normconst,n,1)
end

# centre and normalise data X so that each col has sd=1
function datawhitening(X::Array) 
    for i = 1:size(X,2)   
        X[:,i] = (X[:,i] - mean(X[:,i]))/std(X[:,i])   
    end
    return X
end

#extract features from tensor decomp of each row of X
function feature(X::Array,n::Integer,sigmaRBF::Real,seed::Integer)
    srand(seed)
    N,D=size(X)
    phi=Array(Float64,D,n,N)
    for i=1:N
        Z=randn(D,n)/sigmaRBF
        b=rand(D,n)
        x=repmat(X[i,:],n,1)
        phi[:,:,i]=sqrt(2/n)*cos(x'.*Z+b*2*pi)
    end
    return phi
end

#compute predictions from w,U,I
function pred(w::Array,U::Array,I::Array,phitest::Array)
    D,n,test_size=size(phitest)
    Q=length(w)
    r=size(U,2)
    temp=Array(Float64,D,r,test_size)
    # compute V st V[q,i]=prod_{k=1 to D}(temp[k,I[q,k],i])
    V=Array(Float64,Q,test_size)
    # compute fhat where fhat[i]=V[:,i]'w
    fhat=Array(Float64,test_size)
    for i=1:test_size
        for k=1:D
            temp[k,:,i]=phitest[k,:,i]*U[:,:,k] 
        end
        for q=1:Q
            V[q,i]=prod(diag(temp[:,vec(I[q,:]),i]))
        end
        fhat[i]=dot(V[:,i],w)
    end
    return fhat
end

#plot RMSE over iterations
function RMSE(w_store::Array,U_store::Array,I::Array,phitest::Array,ytest::Array)
    Ntest=length(ytest);
    T=size(w_store,2);
    vecRMSE=Array(Float64,T);
    for i=1:T
        fhat=pred(w_store[:,i],U_store[:,:,:,i],I,phitest);
    	vecRMSE[i]=norm(ytest-fhat)/sqrt(Ntest);
    end
    #plot(vecRMSE)
    return minimum(vecRMSE),indmin(vecRMSE)
end
    

@debug function GPT_SGLDERM(phi::Array,y::Array,sigma::Real,sigma_w::Real,r::Integer,Q::Integer,m::Integer,epsw::Real,epsU::Real,maxepoch::Integer)
    # phi is the D by n by N array of features where phi[k,:,i]=phi^(k)(x_i)
    # sigma is the s.d. of the observed values
    # sigma_w is the s.d. for the Guassian prior on w
    # a is the scale for the Stiefel manifold
    # epsw,epsU are the epsilons for w and U resp.
    # maxepoch is the number of sweeps through whole dataset
    
    D,n,N=size(phi)
    numbatches=int(ceil(N/m))
    
    # initialise w,U^(k)
    w_store=Array(Float64,Q,maxepoch)
    U_store=Array(Float64,n,r,D,maxepoch)
    w=sigma_w*randn(Q)
    #println("w= ",w)
    U=Array(Float64,n,r,D)
    for k=1:D
        Z=randn(r,n)
        U[:,:,k]=transpose(\(sqrtm(Z*Z'),Z)) #sample uniformly from a*V_{n,r}
    end
    
    # fix the random non-zero locations of w
    l=sample(0:(r^D-1),Q,replace=false)
    I=Array(Int32,Q,D)
    for q in 1:Q
        I[q,:]=digits(l[q],r,D)+1
    end
    # this way the locations are drawn uniformly from the lattice [r^D] without replacement
    # so I_qd=index of dth dim of qth non-zero
    
    for epoch=1:maxepoch
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
            # compute V st V[q,i]=prod_{k=1 to D}(temp[k,I[q,k],i])
            V=Array(Float64,Q,batch_size)
            # compute fhat where fhat[i]=V[:,i]'w
            fhat=Array(Float64,batch_size)
            for i=1:batch_size
                for k=1:D
                    temp[k,:,i]=phi_batch[k,:,i]*U[:,:,k] 
                end
                for q=1:Q
                    V[q,i]=prod(diag(temp[:,vec(I[q,:]),i]))
                end
                fhat[i]=dot(V[:,i],w)
            end
	    println("epoch=",epoch," batch=",batch," stdV=",std(V))	
	    #println(V)
	    println("fhat= ",fhat)
	    println("epoch=",epoch," batch=",batch," meanw=",mean(w), " stdw=", std(w))
            # now can compute gradw, the stochastic gradient of log post wrt w
            gradw=((N/batch_size)*V*(y_batch-fhat)-w)/(sigma_w^2)
	    println("epoch=",epoch," batch=",batch," gradw=",gradw)

            # compute U_phi[q,i,k]=expression in big brackets in (11)
            U_phi=Array(Float64,Q,batch_size,D)
            for k=1:D
                U_phi[:,:,k]=V./reshape(temp[k,I[:,k],:],Q,batch_size)
		println("epoch=",epoch," batch=",batch," k=",k," stdU_phi=",std(U_phi[:,:,k]))
            end
            # now compute a_l^(k)(x_i) for l=1,...,r k=1,..,D and store in A
            A=zeros(r,D,batch_size)
            for i=1:batch_size
                for k=1:D
                    for l in unique(I[:,k])
                        index=findin(I[:,k],l) #I_l
                        A[l,:,i]=transpose(reshape(U_phi[index,i,:],length(index),D))*w[index] 
                    end
                end
            end
            # compute Psi as in (12)
            Psi=Array(Float64,n*r,batch_size,D)
            for i=1:batch_size
                for k=1:D
                    Psi[:,i,k]=kron(A[:,k,i],vec(phi_batch[k,:,i]))
                end
            end
	    for k=1:D
		println("epoch=",epoch," batch=",batch," k=",k," stdPsi=",std(Psi[:,:,k]));
	    end
            # can now compute gradU where gradU[:,:,k]=stochastic gradient of log post wrt U^(k)
            gradU=Array(Float64,n,r,D)
            for k=1:D
                gradU[:,:,k]=reshape((N/batch_size)*Psi[:,:,k]*(y[batch]-fhat)/(sigma^2),n,r)
		println("epoch=",epoch," batch=",batch," k=",k," stdgradU=",std(gradU[:,:,k]))
            end

            # SGLD step on w
            w[:]+=epsw*gradw/2 +sqrt(2*epsw)*randn(Q)
	    #println("epoch=",epoch," batch=",batch,"stdw=",std(w))
            # SGLDERM step on U
            for k=1:D
		println("epoch=",epoch," batch=",batch," k=",k," stdUk=",std(U[:,:,k]))
                mom=proj(U[:,:,k],sqrt(epsU)*gradU[:,:,k]/2+randn(n,r),a)
		println("epoch=",epoch," batch=",batch," k=",k," stdmom=",std(mom))
                U[:,:,k]=geod(U[:,:,k],mom,sqrt(epsU),a)
            end
        end
	w_store[:,epoch]=w
	U_store[:,:,:,epoch]=U
    end
    return w_store,U_store,I
end

end
    
    
