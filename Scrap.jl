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

# function to return the negative log marginal likelihood of No Tensor model with Gaussian observations
# hyperparams include signal_var, which should always be hyperparams[end]
# randfeature is the function with arg hyperparams generating random features for the No Tensor model. It should ignore signal_var=hyperparams[end]
# random_numbers are the random numbers used to generate features in randfeature. This can be an array or a Tuple
function GPNT_nlogmarginal(y::Array,n::Integer,hyperparams::Vector,randfeature::Function)
	N=length(y);
    phi=randfeature(hyperparams);
	signal_var=hyperparams[end];
    A=phi*phi'+signal_var*eye(n);
	Chol=cholfact(A);L=Chol[:L]; U=Chol[:U] # L*U=A
    b=phi*y;
	l=\(U,\(L,b)); #inv(A)*phi*y
	logdetA=2*sum(log(diag(L)));
    return (N-n)*log(signal_var)/2+logdetA/2+(sum(y.*y)-sum(b.*l))/(2*signal_var)
end

# function to return the gradient of negative log marginal likelihood of No Tensor model with Gaussian observations
# hyperparams include signal_var, which should always be hyperparams[end]
# gradfeature is a function with args hyperparams giving the gradient of random features for the No Tensor model wrt hyperparams. It should ignore signal_var=hyperparams[end], and give an n by N by Lh array where Lh=length(hyperparams)
# function should return a vector of length Lh
function GPNT_gradnlogmarginal(y::Array,n::Integer,hyperparams::Vector,randfeature::Function,gradfeature::Function)
	N=length(y);
	phi=randfeature(hyperparams);
	signal_var=hyperparams[end];
	A=phi*phi'+signal_var*eye(n);
	Chol=cholfact(A); L=Chol[:L]; U=Chol[:U] # L*U=A
	gradphi=gradfeature(hyperparams);
	Lh=length(hyperparams);
	signal_var=hyperparams[Lh];
	grad=Array(Float64,Lh);
	b=phi*y
	l=\(U,\(L,b)); #inv(A)*phi*y
	for h=1:(Lh-1)
		gphi=gradphi[:,:,h]; #dphi/dh
		temp=\(U,\(L,gphi));	#inv(A)*dphi/dh
		B=phi'*temp; #phi'*inv(A)*dphi/dh
		c=phi'*l-y; #(phi'*inv(A)*phi-I)y
		grad[h]=trace(B)+sum(y.*(B*c))/signal_var
	end
	lambda=eigvals(A);
	grad[Lh]=(N-n)/(2*signal_var)+sum(1./lambda)/2+(sum(l.*(phi*y))-sum(y.^2))/(2*signal_var^2)+sum(l.^2)/(2*signal_var)
	return grad
end


# learning hyperparams by optimising Gaussian marginal likelihood wrt positive hyperparams
# hyperparams include signal_var, which should always be hyperparams[end]
# nlogmarginal is the negative log marginal lkhd, a function with input argument hyperparams only
# gradnlogmarginal is the gradient of nlogmarginal wrt hyperparams
function GPNT_hyperparameters(nlogmarginal::Function,gradnlogmarginal::Function,init_hyperparams::Vector)
nlm(loghyperparams::Vector)=nlogmarginal(exp(loghyperparams)); # exp needed to enable unconstrained optimisation, since hyperparams must be positive
g(loghyperparams::Vector)=gradnlogmarginal(exp(loghyperparams)).*exp(loghyperparams)
    function g!(loghyperparams::Vector,storage::Vector)
        grad=g(loghyperparams)
        for i=1:length(loghyperparams)
            storage[i]=grad[i]
        end
    end
    l=optimize(logmarginal,g!,log(init_hyperparams),method=:cg,show_trace = true, extended_trace = true)
	return exp(l.minimum)
end

# extract random fourier features from tensor decomp of each row of X
# set fixed seed using srand(seed) to use the same Z and b
# fixedl=true if using same length_scale for all dimensions. o/w false.
function RFFtensor(X::Array,n::Integer,hyperparams::Vector,phi_scale::Real,Z::Array=randn(n,size(X,2)),b::Array=2*pi*rand(n);fixedl=true)
	N,D=size(X);
	if fixedl
		length_scale=hyperparams[1]; sigma_RBF=hyperparams[2];
	else length_scale=hyperparams[1:D]; sigma_RBF=hyperparams[D+1];
	end
	phi=Array(Float64,n,D,N)
	Zt=scale(Z,1./length_scale)
	for i=1:N
		for k=1:D
			for j=1:n
				phi[j,k,i]=cos(X[i,k]*Zt[j,k]+b[j,k])
			end
		end
    end
    return phi_scale*(sigma_RBF)^(1/D)*sqrt(2/n)*phi
end

# alternative Fourier feature embedding
# set fixed seed using srand(seed) to use the same Z
# n must be even
function RFFtensor2(X::Array,n::Integer,hyperparams::Vector,phi_scale::Real,Z::Array=randn(int(n/2),D);fixedl=true)    
	if n%2==1
		error("n not even")
	end
	half_n=int(n/2);
	N,D=size(X)
	if fixedl
		length_scale=hyperparams[1]; sigma_RBF=hyperparams[2];
	else length_scale=hyperparams[1:D]; sigma_RBF=hyperparams[D+1];
	end
	phi=Array(Float64,n,D,N)
	Zt=scale(Z,1./length_scale)
	for i=1:N
		for k=1:D
			for j=1:half_n
				phi[2*j-1,k,i]=sin(X[i,k]*Zt[j,k])
				phi[2*j,k,i]=cos(X[i,k]*Zt[j,k])
			end
		end
	end
	return phi_scale*(sigma_RBF)^(1/D)*phi/sqrt(half_n)
end

# random fourier feature embedding for the no tensor model (full-theta)
# set fixed seed using srand(seed) to use the same Z and b
# fixedl=true if using same length_scale for all dimensions. o/w false.
function RFF(X::Array,n::Integer,hyperparams::Vector,Z::Array=randn(n,size(X,2)),b::Array=2*pi*rand(n);fixedl=true)    
    N,D=size(X);
	if fixedl
		length_scale=hyperparams[1]; sigma_RBF=hyperparams[2];
	else length_scale=hyperparams[1:D]; sigma_RBF=hyperparams[D+1];
	end
    phi=Array(Float64,n,N)
    Zt=scale(Z,1./length_scale)
    for i=1:N
		for j=1:n
        	phi[j,i]=cos(sum(X[i,:].*Zt[j,:]) + b[j])
		end
    end
    return sqrt(2/n)*sigma_RBF*phi
end

# alternative fourier feature embedding for the no tensor model (full-theta)
# set fixed seed using srand(seed) to use the same Z
# fixedl=true if using same length_scale for all dimensions. o/w false.
function RFF2(X::Array,n::Integer,hyperparams::Vector,Z::Array=randn(int(n/2),D);fixedl=true) 
	if n%2==1
		error("n not even")
	end
	half_n=int(n/2);
	N,D=size(X)
	if fixedl
		length_scale=hyperparams[1]; sigma_RBF=hyperparams[2];
	else length_scale=hyperparams[1:D]; sigma_RBF=hyperparams[D+1];
	end
	phi=Array(Float64,n,N)
	Zt=scale(Z,1./length_scale)
	for i=1:N
	    for j=1:half_n
	        temp=sum(X[i,:].*Zt[j,:])
	        phi[2*j-1,i]=sin(temp)
	        phi[2*j,i]=cos(temp)
	    end                    
	end
	return sigma_RBF*phi/sqrt(half_n)
end

# function to give grad of RFF wrt length_scale and sigma_RBF
# set fixed seed using srand(seed) to use the same Z and b
# returns an n by N by Lh array where Lh=length(hyperparams), where [:,:,h] is the gradient of RFF wrt hyperparameter h
function gradRFF(X::Array,n::Integer,hyperparams::Vector,Z::Array=randn(n,size(X,2)),b::Array=2*pi*rand(n);fixedl=true)
    N,D=size(X);
	features=Array(Float64,n,N)
	if fixedl
		length_scale=hyperparams[1]; sigma_RBF=hyperparams[2];
		Zt=scale(Z,1./length_scale)
		for i=1:N
			for j=1:n
			   	features[j,i]=sum(X[i,:].*Zt[j,:]) + b[j]
			end
		end
		phisin=sqrt(2/n)*sigma_RBF*sin(features);
		return cat(3,phisin.*(Zt*X')/length_scale,sqrt(2/n)*cos(features))		
	else length_scale=hyperparams[1:D]; sigma_RBF=hyperparams[D+1];
		gradl=Array(Float64,n,N,D)
		Zt=scale(Z,1./length_scale)
		for i=1:N
			for j=1:n
			   	features[j,i]=sum(X[i,:].*Zt[j,:]) + b[j]
			end
		end
		phisin=sqrt(2/n)*sigma_RBF*sin(features);
		for k=1:D
		    gradl[:,:,k]=phisin.*(Zt[:,k]*X[:,k]')/length_scale[k]
		end
		return cat(3,gradl,sqrt(2/n)*cos(features))
	end
end

function RFFtest(X::Array,n::Integer,hyperparams::Vector,Z,b) 
    N,D=size(X);
	length_scale=hyperparams[1]; sigma_RBF=hyperparams[2];
    phi=Array(Float64,n,N)
    Zt=scale(Z,1./length_scale)
    for i=1:N
		for j=1:n
        	phi[j,i]=cos(sum(X[i,:].*Zt[j,:]) + b[j])
		end
    end
    return sqrt(2/n)*sigma_RBF*phi
end

