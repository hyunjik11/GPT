for i=1:10
    subplot(2,5,i)
    ax=gca()
    plot(vec(theta_store[10*i,:]))
    plot(vec(theta_store2[10*i,:]))
    ylim(-5,5)
    xlabel("num_iterations(no burnin)")
    setp(ax[:get_xticklabels](),fontsize=8)
end

# Tensor Regression Model
function GPTregression(phi::Array, y::Array, signal_var::Real, I::Array, r::Integer, Q::Integer, m::Integer, epsw::Real, epsU::Real, burnin::Integer, maxepoch::Integer;langevin=true,stiefel=true)
    # phi is the D by n by N array of features where phi[k,:,i]=phi^(k)(x_i)
	# phi should have been constructed using scale sqrt(n/(Q^(1/D)))
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
	
	if stiefel
		U=Array(Float64,n,r,D)
		for k=1:D
		    Z=randn(r,n)
		    U[:,:,k]=transpose(\(sqrtm(Z*Z'),Z)) #sample uniformly from V_{n,r}
		end
	else U=randn(n,r,D)/sqrt(n)
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
	    
            # update w
			if langevin
				w+=epsw*gradw/2+sqrt(epsw)*randn(Q)
			else w+=epsw*gradw/2
			end

            # update U
			if langevin
				if stiefel
				    for k=1:D
				        mom=proj(U[:,:,k],sqrt(epsU)*gradU[:,:,k]/2+randn(n,r))
				        U[:,:,k]=geod(U[:,:,k],mom,sqrt(epsU));
				        if U[:,:,k]==zeros(n,r) #if NaN appears while evaluating G
				            return zeros(Q,maxepoch*numbatches),zeros(n,r,D,maxepoch*numbatches)
				        end
				    end
				else U+=epsw_gradU/2+sqrt(epsU)*randn(n,r,D)
				end
			else
				if stiefel
					for k=1:D
				        mom=proj(U[:,:,k],sqrt(epsU)*gradU[:,:,k]/2)
				        U[:,:,k]=geod(U[:,:,k],mom,sqrt(epsU));
				        if U[:,:,k]==zeros(n,r) #if NaN appears while evaluating G
				            return zeros(Q,maxepoch*numbatches),zeros(n,r,D,maxepoch*numbatches)
				        end
				    end
				else U+=epsw_gradU/2
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

# Tensor multi-class classification Model
# y must be a vector of integer labels in {1,...,C}
function GPTclassification(phi::Array, y::Array, I::Array, r::Integer, Q::Integer, m::Integer, epsw::Real, epsU::Real, burnin::Integer, maxepoch::Integer;langevin=true,stiefel=true)
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
	
	if stiefel
		U=Array(Float64,n,r,D,C)
		for k=1:D
			for c=1:C
		    	Z=randn(r,n)
		    	U[:,:,k,c]=transpose(\(sqrtm(Z*Z'),Z)) #sample uniformly from V_{n,r}
			end
		end
	else U=randn(n,r,D,C)
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
			gradwlogsumexpfhat_c=Array(Float64,Q,batch_size,C)
			for i=1:batch_size
				for c=1:C
					gradwlogsumexpfhat_c[:,i,c]=exp(fhat[i,c]-tmp[i])*V[:,i,c]
				end
			end

            # now can compute gradw, the stochastic gradient of log post wrt w
			gradw=-squeeze(sum(gradwlogsumexpfhat_c,2),2)
			for i=1:batch_size
            		gradw[:,y_batch[i]]+=V[:,i,y_batch[i]]
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
			gradUlogsumexpfhat_c=Array(Float64,n*r,batch_size,D,C)
			for i=1:batch_size
				for k=1:D
					for c=1:C
						gradUlogsumexpfhat_c[:,i,k,c]=exp(fhat[i,c]-tmp[i])*Psi[:,i,k,c]
					end
				end
			end

            # can now compute gradU where gradU[:,:,k]=stochastic gradient of log post wrt U^(k)
            gradU=-squeeze(sum(gradUlogsumexpfhat_c,2),2)
            for i=1:batch_size
                gradU[:,:,y_batch[i]]+=squeeze(Psi[:,i,:,y_batch[i]],2)
            end
			gradU=N/batch_size*reshape(gradU,n,r,D,C);
			#=
			#check gradients
			if batch==1
				#sample random entry of w
				dw=1e-1; dU=1e-2;
				myq=sample(1:Q,1); myc=sample(1:C,1);
				pt=zeros(Q,C); pt[myq,myc]+=dw;
				wperturbed=w+pt;
			
				# compute fhat where fhat[i,c]=V[:,i,c]'w[:,c]
				fhatw=Array(Float64,batch_size,C)
				for c=1:C
		        	fhatw[:,c]=computefhat(V[:,:,c],wperturbed[:,c])
				end
			
				#compute logsumexp(fhat[i,:]) and store as tmp[i]
				tmpw=Array(Float64,batch_size)
				for i=1:batch_size
					tmpw[i]=logsumexp(fhatw[i,:])
				end
			
				myn=sample(1:n,1); myr=sample(1:r,1); myk=sample(1:D,1); 
				pt=zeros(n,r,D,C); pt[myn,myr,myk,myc]+=dU;
				Uperturbed=U+pt
				# compute <phi^(k)(x_i),U^(c,k)_{.l}> for all k,l,batch,c and store in temp
				tempU=Array(Float64,D,r,batch_size,C)            
				for c=1:C
					tempU[:,:,:,c]=phidotU(Uperturbed[:,:,:,c],phi_batch)
				end

				# compute V st V[q,i,c]=prod_{k=1 to D}(temp[k,I[q,k],i,c])
				VU=Array(Float64,Q,batch_size,C)
				for c=1:C
			        VU[:,:,c]=computeV(tempU[:,:,:,c],I)
				end
						
		        # compute fhat where fhat[i,c]=V[:,i,c]'w[:,c]
				fhatU=Array(Float64,batch_size,C)
				for c=1:C
		        	fhatU[:,c]=computefhat(VU[:,:,c],w[:,c])
				end
			
				#compute logsumexp(fhat[i,:]) and store as tmp[i]
				tmpU=Array(Float64,batch_size)
				for i=1:batch_size
					tmpU[i]=logsumexp(fhatU[i,:])
				end

				l=0;lw=0;lU=0;
				for i=1:batch_size
					l+=fhat[i,y_batch[i]]-tmp[i]
					lw+=fhatw[i,y_batch[i]]-tmpw[i]
					lU+=fhatU[i,y_batch[i]]-tmpU[i]
				end
			
				println("gradwentry=",(batch_size/N)*(gradw[myq,myc]+w[myq,myc]/(sigma_w^2))," fdgradw=",(lw-l)/dw)
				println("gradUentry=",(batch_size/N)*gradU[myn,myr,myk,myc]," fdgradU=",(lU-l)/dU)
			end
			=#
            # SGLD step on w
            w+=epsw*gradw/2 +sqrt(epsw)*randn(Q,C)

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

            # update w
			if langevin
				w+=epsw*gradw/2+sqrt(epsw)*randn(Q,C)
			else w+=epsw*gradw/2
			end

            # update U
			if langevin
				if stiefel
				    for k=1:D
						for c=1:C
						    mom=proj(U[:,:,k,c],sqrt(epsU)*gradU[:,:,k,c]/2+randn(n,r))
						    U[:,:,k,c]=geod(U[:,:,k,c],mom,sqrt(epsU));
						    if U[:,:,k,c]==zeros(n,r) #if NaN appears while evaluating G
						        return zeros(Q,C,maxepoch*numbatches),zeros(n,r,D,C,maxepoch*numbatches)
						    end
						end
				    end
				else U+=epsw_gradU/2+sqrt(epsU)*randn(n,r,D,C)
				end
			else
				if stiefel
					for k=1:D
						for c=1:C
						    mom=proj(U[:,:,k,c],sqrt(epsU)*gradU[:,:,k,c]/2)
						    U[:,:,k,c]=geod(U[:,:,k,c],mom,sqrt(epsU));
						    if U[:,:,k,c]==zeros(n,r) #if NaN appears while evaluating G
						        return zeros(Q,C,maxepoch*numbatches),zeros(n,r,D,C,maxepoch*numbatches)
						    end
						end
				    end
				else U+=epsw_gradU/2
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

