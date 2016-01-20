if 1==0
c=h5open("theta.h5","w") do file
    write(file,"theta_store",theta_store)
    write(file,"theta_store2",theta_store2)
end

file="theta.h5";
theta_store=h5read(file,"theta_store");
theta_store2=h5read(file,"theta_store2");

figure()

for i=1:10
    subplot(2,5,i)
    ax=gca()
    plot(vec(theta_store[10*i,:]))
    plot(vec(theta_store2[10*i,:]))
    ylim(-5,5)
    xlabel("num_iterations(no burnin)")
    setp(ax[:get_xticklabels](),fontsize=8)
end

savefig("/homes/hkim/GPT/Plots/NoTensorSGLDThetaTraceDecay")
end

using ReverseDiffSource
using Optim

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
            gradw=(N/batch_size)*V*(y_batch-fhat)/signal_var-w/(sigma_w^2)

            # update gw and compute step size for w
            gw=alpha*gw+(1-alpha)*(gradw.^2) 
            epsw=epsilon./(sqrt(gw)+lambda)
            println("norm of gw=",sqrt(sum(gw.^2)),";std of gw=",std(gw))
            println("norm of epsw=",sqrt(sum(epsw.^2)),";std of epsw=",std(epsw))
            
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
            println("norm of gU=",sqrt(sum(gU.^2)),";std of gU=",std(gU))
            println("norm of meanepsU=",sqrt(sum(meanepsU.^2)),";std of meanepsU=",std(meanepsU))

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


    

