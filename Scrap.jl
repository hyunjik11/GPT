for i=1:10
    subplot(2,5,i)
    ax=gca()
    plot(vec(theta_store[10*i,:]))
    plot(vec(theta_store2[10*i,:]))
    ylim(-5,5)
    xlabel("num_iterations(no burnin)")
    setp(ax[:get_xticklabels](),fontsize=8)
end

# SGLD for multi-class classification on Tucker Model with Stiefel Manifold
# y must be a vector of integer labels in {1,...,C}
function GPT_SGLDERMclass(phi::Array, y::Array, I::Array, r::Integer, Q::Integer, m::Integer, epsw::Real, epsU::Real, burnin::Integer, maxepoch::Integer)
    # phi is the D by n by N array of features where phi[k,:,i]=phi^(k)(x_i)
    # epsw,epsU are the epsilons for w and U resp.
    # maxepoch is the number of sweeps through whole dataset
    
    n,D,N=size(phi)
    numbatches=int(ceil(N/m))
    sigma_w=1;
	C=int(maximum(y)-minimum(y)+1)
    
    # initialise w,U^(k)
    w_store=Array(Float64,Q,C,maxepoch*numbatches)
    U_store=Array(Float64,n,r,D,C,maxepoch*numbatches)
    w=sigma_w*randn(Q,C)

    U=Array(Float64,n,r,D,C)
    for k=1:D
		for c=1:C
        	Z=randn(r,n)
        	U[:,:,k,c]=transpose(\(sqrtm(Z*Z'),Z)) #sample uniformly from V_{n,r}
		end
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

            # compute <phi^(k)(x_i),U^(c,k)_{.l}> for all k,l,batch,c and store in temp
			temp=Array(Float64,D,r,batch_size,C)            
			for c=1:C
				temp[:,:,:,c]=phidotU(U[:,:,:,c],phi_batch)
			end

	    	# compute V st V[q,i,c]=prod_{k=1 to D}(temp[k,I[q,k],i,c])
			V=Array(Float64,Q,batch_size,C)
			for c=1:C
	            V[:,:,c]=computeV(temp[:,:,:,c],I)
			end
				    
            # compute fhat where fhat[i,c]=V[:,i,c]'w[:,c]
			fhat=Array(Float64,batch_size,C)
			for c=1:C
            	fhat[:,c]=computefhat(V[:,:,c],w[:,c])
			end
			
			#compute logsumexp(fhat[i,:]) and store as tmp[i]
			tmp=Array(Float64,batch_size)
			for i=1:batch_size
				tmp[i]=logsumexp(fhat[i,:])
			end

			# compute gradwlogsumexpfhat_c[:,i]=gradw(log(sum_c(fhat[i,c])))
			gradwlogsumexpfhat_c=zeros(Q,batch_size)
			for i=1:batch_size
				for c=1:C
					gradwlogsumexpfhat_c[:,i]+=exp(fhat[i,c]-tmp[i])*V[:,i,c]
				end
			end

            # now can compute gradw, the stochastic gradient of log post wrt w
			gradw=Array(Float64,Q,C)
			for c=1:C
            	gradw[:,c]=vec(sum(V[:,:,c]-gradwlogsumexpfhat_c,2))
			end
			gradw*=N/batch_size
			gradw-=w/(sigma_w^2)

            # compute U_phi[q,i,k,c]=expression in big brackets in (11)
			U_phi=Array(Float64,Q,batch_size,D,C)
			for c=1:C
            	U_phi[:,:,:,c]=computeU_phi(V[:,:,c],temp[:,:,:,c],I)
			end
            
            # compute a_l^(k)(x_i) for l=1,...,r k=1,..,D and store in A
			A=Array(Float64,r,D,batch_size,C)            
			for c=1:C
				A[:,:,:,c]=computeA(U_phi[:,:,:,c],w[:,c],I,r)
			end
            
            # compute Psi as in (12)
			Psi=Array(Float64,n*r,batch_size,D,C)
			for c=1:C            
				Psi[:,:,:,c]=computePsi(A[:,:,:,c],phi_batch)
			end

			# compute gradUlogsumexpfhat_c[:,i]=gradU(log(sum_c(fhat[i,c])))
			gradUlogsumexpfhat_c=zeros(n*r,D,batch_size)
			for i=1:batch_size
				for c=1:C
					gradUlogsumexpfhat_c[:,k,i]+=exp(fhat[i,c]-tmp[i])*Psi[:,i,k,c]
				end
			end

            # can now compute gradU where gradU[:,:,k]=stochastic gradient of log post wrt U^(k)
            gradU=Array(Float64,n,r,D,C)
            for k=1:D
				for c=1:C
                	gradU[:,:,k,c]=reshape(squeeze(sum(Psi[:,:,k,c],2),[2,3,4])-squeeze(sum(gradUlogsumexpfhat_c[:,k,:],3),[2,3]),n,r)
				end
            end
			gradU*=N/batch_size
	    
            # SGLD step on w
            w+=epsw*gradw/2 +sqrt(epsw)*randn(Q,C)
	    #if batch==1
	    #	println("mean epsgradw_half=",mean(epsw*gradw/2)," std =",std(epsw*gradw/2))
	    #	println("meansqrtepsgradU_half=",mean(sqrt(epsU)*gradU/2), " std=",std(sqrt(epsU)*gradU/2))
	    #end
            # SGLDERM step on U
            for k=1:D
				for c=1:C
                	mom=proj(U[:,:,k,c],sqrt(epsU)*gradU[:,:,k,c]/2+randn(n,r))
                	U[:,:,k,c]=geod(U[:,:,k,c],mom,sqrt(epsU));
                	if U[:,:,k,c]==zeros(n,r) #if NaN appears while evaluating G
                    	return zeros(Q,C,maxepoch*numbatches),zeros(n,r,D,C,maxepoch*numbatches)
                	end
            	end
			end
	    	if epoch>burnin
	        		w_store[:,:,((epoch-burnin)-1)*numbatches+batch]=w
	        		U_store[:,:,:,:,((epoch-burnin)-1)*numbatches+batch]=U
	    	end
        end
    end
    return w_store,U_store
end
