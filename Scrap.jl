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

#log marginal likelihood of No Tensor Model
function GPNT_logmarginal(X::Array,y::Array,n::Integer,length_scale::Real,sigma_RBF::Real,sigma::Real,seed::Integer)
    phi=featureNotensor(X,n,length_scale,sigma_RBF,seed);
    A=phi*phi'+sigma^2*I;
    b=phi*y;
    B=\(A,b);
    return (n-N)*log(sigma)-log(det(A))/2-(sum(y.*y)-b'*B)/(2*sigma^2)
end

#learning hyperparams sigma,sigma_RBF,length_scale for No Tensor Model by optimising marginal likelihood
function GPNT_hyperparameters(X::Array,y::Array,n::Integer,init_length_scale::Real,init_sigma_RBF::Real,init_sigma::Real,seed::Integer)
    logmarginal(length_scale,sigma_RBF,sigma)=GPNT_logmarginal(X,y,n,length_scale,sigma_RBF,sigma,seed); # log marginal likelihood as a fn of length_scale,sigma_RBF,sigma only.
    optimize(logmarginal,[init_length_scale,init_sigma_RBF,init_sigma],method=:cg,autodiff=true)
end

    
    

