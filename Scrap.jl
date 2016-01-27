for i=1:10
    subplot(2,5,i)
    ax=gca()
    plot(vec(theta_store[10*i,:]))
    plot(vec(theta_store2[10*i,:]))
    ylim(-5,5)
    xlabel("num_iterations(no burnin)")
    setp(ax[:get_xticklabels](),fontsize=8)
end

#input theta_vector is a vector of length n*C - converted to n by C array within function
function neglogjointlkhd(theta_vec::Vector,hyperparams::Vector)
	if length(hyperparams)==2 # if length_scale is a scalar	
		length_scale=hyperparams[1]
	else length_scale=hyperparams[1:end-1]
	end
	sigma_RBF=hyperparams[end]
	theta=reshape(theta_vec,n,C) # n by C matrix
    phi=featureNotensor(Xtrain,n,length_scale,sigma_RBF,seed) # n by Ntrain matrix
    phi_theta=phi'*theta # Ntrain by C matrix
    exp_phi_theta=exp(phi_theta) 
    L=0;
    for i=1:Ntrain
        L+=log(sum(exp_phi_theta[i,:]))-phi_theta[i,ytrain[i]]
    end
    L+=sum(abs2(theta))/2
    return L 
end

function gradneglogjointlkhd(theta_vec::Vector,hyperparams::Vector)
	theta=reshape(theta_vec,n,C) # n by C matrix
	sigma_RBF=hyperparams[end]	

	if length(hyperparams)==2 # if length_scale is a scalar
		length_scale=hyperparams[1]	
		phi=featureNotensor(Xtrain,n,length_scale,sigma_RBF,seed) # n by Ntrain matrix
		exp_phi_theta=exp(phi'*theta)	# Ntrain by C matrix
		sum_exp_phi_theta=sum(exp_phi_theta,2) # Vector length Ntrain
		gradtheta=zeros(n,C)
		for c=1:C
			for i=1:Ntrain
				gradtheta[:,c]+=exp_phi_theta[i,c]*phi[:,i]/sum_exp_phi_theta[i]
			end
		end
		for i=1:Ntrain
			gradtheta[:,ytrain[i]]-=phi[:,i]
		end
		gradtheta+=theta

		gradfeature=gradfeatureNotensor(Xtrain,n,length_scale,sigma_RBF,seed)
		gradlength_scale=0;
		gradsigma_RBF=0;
		for i=1:Ntrain
			gradlength_scale+=sum(exp_phi_theta[i,:].*(theta'*gradfeature[1][:,i]))/sum_exp_phi_theta[i]-sum(theta[:,ytrain[i]].*gradfeature[1][:,i]')
			gradsigma_RBF+=sum(exp_phi_theta[i,:].*(theta'*gradfeature[2][:,i]))/sum_exp_phi_theta[i]-sum(theta[:,ytrain[i]].*gradfeature[2][:,i]')
		end
	else # length_scale is a vector - varying length scales across input dimensions
		length_scale=hyperparams[1:end-1] 
		phi=featureNotensor(Xtrain,n,length_scale,sigma_RBF,seed) # n by Ntrain matrix
		exp_phi_theta=exp(phi'*theta)	# Ntrain by C matrix
		sum_exp_phi_theta=sum(exp_phi_theta,2) # Vector length Ntrain
		gradtheta=zeros(n,C)
		for c=1:C
			for i=1:Ntrain
				gradtheta[:,c]+=exp_phi_theta[i,c]*phi[:,i]/sum_exp_phi_theta[i]
			end
		end
		for i=1:Ntrain
			gradtheta[:,ytrain[i]]-=phi[:,i]
		end
		gradtheta+=theta
	
		gradfeature=gradfeatureNotensor(Xtrain,n,length_scale,sigma_RBF,seed)
		gradlength_scale=zeros(D);
		gradsigma_RBF=0;
		for i=1:Ntrain
			dphi_xibydl=squeeze(gradfeature[1][:,i,:],2)
			gradlength_scale+=(exp_phi_theta[i,:]*theta'*dphi_xibydl-theta[:,ytrain[i]]'*dphi_xibydl)'
			gradsigma_RBF+=sum(exp_phi_theta[i,:].*(theta'*gradfeature[2][:,i]))/sum_exp_phi_theta[i]-sum(theta[:,ytrain[i]].*gradfeature[2][:,i]')
		end
	end
	return [vec(gradtheta),gradlength_scale,gradsigma_RBF]
end


