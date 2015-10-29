#implementation with batches
module GPT_SGLD

using Distributions

export proj,geod,GPT_SGLDERM
    
# define proj for Stiefel manifold
function proj(U,V)
    return V-U*(U'*V+V'*U)/2
end

# define geod for Stiefel manifold
function geod(U,mom,t)
    A=U'*mom
    temp=[A -mom'*mom;eye(r) A]
    E=exp(t*temp)
    return [U mom]*E[:,1:r]*exp(-t*A)
end

function GPT_SGLDERM(phi,y,sigma,sigma_w,r,Q,m,epsw,epsU,maxepoch)
    # phi is the D by n by N array of features where phi[k,:,i]=phi^(k)(x_i)
    # sigma is the s.d. of the observed values
    # sigma_w is the s.d. for the Guassian prior on w
    # epsw,epsU are the epsilons for w and U resp.
    # maxepoch is the number of sweeps through whole dataset
    
    D,n,N=size(phi)
    numbatches=ceil(N/m)
    
    # initialise w,U^(k)
    w_store=Array(Float64,Q,numbatches*maxepoch)
    U_store=Array(Float64,n,r,D,numbatches*maxepoch)
    w=sigma_w*randn(Q)
    U=Array(Float64,n,r,D)
    for k=1:D
        Z=randn(r,n)
        U[:,:,k]=transpose(Z/sqrtm(Z*Z')) #sample uniformly from V_{n,r}
    end
    
    # fix the random non-zero locations of w
    I=rand(DiscreteUniform(1, r),Q,D) 
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
                fhat[i]=transpose(V[:,i])*w
            end

            # now can compute gradw, the stochastic gradient of log post wrt w
            gradw=((N/batch_size)*V*(y_batch-fhat)-w)/(2*sigma_w^2)

            # compute U_phi[q,i,k]=expression in big brackets in (11)
            U_phi=Array(Float64,Q,batch_size,D)
            for k=1:D
                U_phi[:,:,k]=V./reshape(temp[k,I[:,k],:],Q,batch_size)
            end

            # now compute a_l^(k)(x_i) for l=1,...,r k=1,..,D and store in A
            A=zeros(r,D,batch_size)
            for i=1:batch_size
                for l in unique(I[:,k])
                    index=findin(I[:,k],l) #I_l
                    A[l,:,i]=transpose(reshape(U_phi[index,i,:],length(index),D))*w[index] 
                end
            end

            # compute Psi as in (12)
            Psi=Array(Float64,n*r,batch_size,D)
            for i=1:batch_size
                for k=1:D
                    Psi[:,i,k]=kron(A[:,k,i],vec(phi_batch[k,:,i]))
                end
            end

            # can now compute gradU where gradU[:,:,k]=stochastic gradient of log post wrt U^(k)
            gradU=Array(Float64,n,r,D)
            for k=1:D
                gradU[:,:,k]=reshape((N/batch_size)*Psi[:,:,k]*(y[batch]-fhat)/(2*sigma^2),n,r)
            end

            # SGLD step on w
            w[:]+=epsw*gradw/2 +sqrt(2*epsw)*randn(Q)
            w_store[:,numbatches*(epoch-1)+batch]=w
            # SGLDERM step on U
            for k=1:D
                mom=proj(U[:,:,k],sqrt(epsU)*gradU[:,:,k]/2+randn(n,r))
                U[:,:,k]+=geod(U[:,:,k],mom,sqrt(epsU))
                U_store[:,:,k,numbatches*(epoch-1)+batch]=U[:,:,k]
            end
        end
    end
    return w_store,U_store,I
end

end
    
    
