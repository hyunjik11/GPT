module GaussianProcess

using PyPlot


export GP, GPmean, GPcov, SECov, Mean, Cov, GPrand, GPpost, plot1d

type GP
    mean::Union(Function,Real) #mean function (could also be const)
    cov::Function #covariance function
    dim::Integer #ndims of the domain of GP
end

function GPmean(gp::GP)
    #function to return mean function of GP
    return gp.mean
end

function GPcov(gp::GP)
    #function to return covariance function of GP
    return gp.cov
end


######### List of covariance functions ###############
function SECov(length::Real,sigma::Real)
    function f(x,y)
        return sigma^2*exp(-norm(x-y)^2/(2*length^2))
    end
    return f
end

######################################################

function Mean(gp::GP,x_values)
    #function to return gp.mean(x_values) and gp.cov(x_values,x_values)
    #mean as 1D float array and cov as float 2D array
    #Can assume input x_values is a 2D array
    n=size(x_values,1); #number of x_values
    
    #first evaluate mean
    meanfn=gp.mean;
    if typeof(meanfn)<:Real #deals with case where mean is const
        mean=[meanfn for i=1:n];
    else #if mean is a function
        mean=[meanfn(x_values[i,:]) for i=1:n];
    end
    m=float(map(scalar,mean));
    return m
end

function Cov(gp::GP,x_values)
    n=size(x_values,1);
    covfn=gp.cov;
    K=[covfn(x_values[i,:],x_values[j,:]) for i=1:n, j=1:n];
    return float(K)
end

function GPrand(gp::GP,x_values)
    #Function to sample a random function from GP, 
    #and evaluate the function at the specified x_values.
    temp=convertTo2D(x_values);
    x_values=dimensionCheck(gp.dim,temp);
    
    n=size(x_values,1);
    m=Mean(gp,x_values);
    K=Cov(gp,x_values);
    R=chol(K); #R'*R=cov
    z=randn(n);
    return R'*z+m
end

function GPpost(gp::GP,x_values,y_values,sigma_noise::Real)
    #function to return the posterior GP conditioned on
    #input x_values where each x_value corresponds to a row of array
    #output y_values under the model y~N(f(x),sigma_noise^2)
    #need y_values to be a 1D array
    temp=convertTo2D(x_values);
    x_values=dimensionCheck(gp.dim,temp);
    temp=convertTo2D(y_values);
    y_values=vec(temp); #want to keep as 1D array
    
    n=size(x_values,1);
    m=Mean(gp,x_values);
    K=Cov(gp,x_values);
    meanfn=gp.mean;
    covfn=gp.cov;
    
    #define function covfn(x,x_values) which returns 1D array
    function k(x)
        temp2=convertTo2D(x);
        a=float([covfn(temp2,x_values[i,:]) for i=1:n]);
        return a
    end
    
    function mean_post(x)
	temp3=\(K+sigma_noise^2*eye(n),y_values-m)
	mean=dot(k(x),temp3)
        if typeof(meanfn)<:Real #when gp.mean is const
            temp4=meanfn+mean
            return scalar(temp4)
        else
            temp4=meanfn(x)+mean
            return scalar(temp4)
        end
    end
    
    function cov_post(x,y)
	temp5=\(K+sigma_noise^2*eye(n),k(y))
	cov=covfn(x,y)-k(x)'*temp5
        return scalar(cov)
    end
    
    return GP(mean_post,cov_post,gp.dim)
end   
    
function plot1d(gp::GP,xrange=[-1,1])
    #function to plot the mean of GP in xrange 
    #with 95% confidence intervals for gp.dim=1
    if gp.dim==1
        x=linspace(xrange[1],xrange[2],101);
        m=Mean(gp,reshape(x,length(x),1));
	K=Cov(gp,reshape(x,length(x),1));
        cov=diag(K);
        s=sqrt(cov);
        plot(x,m,"r-");
        fill_between(x,m-2s,m+2s,color="0.75")        
    else
        error("GP dimension not equal to 1")
    end
end

################### private functions #################
function convertTo2D(x)
    #function to convert objects of dimension 0,1 to 2D
    p=ndims(x);
    if p>2
        error ("dimension of x_values exceeds 3")
    end
    if p==0
        x_new=reshape([x],1,1);
        return x_new
    elseif p==1
        #when dim=1, default to row vector
        x_new=reshape(x,1,length(x));
        return x_new
    else 
        return x
    end
end
    
function dimensionCheck(gpdim,x_values)
    #function to check that dimensions of x values=gpdim
    #Can Assume that x_values is a 2D array.
    #Say size(x_values)=(n,p).
    #We also need to distinguish between cases:
    #1. x_values is a single point in dim>1
    #2. x_values is multiple points in 1D
    (n,p)=size(x_values);
    if p==1 && gpdim==n && n>1 #case 1 but have col vector
        x_new=x_values';
    elseif n==1 && gpdim==1 && p>1 #case 2 but have row vector
        x_new=x_values';
    else    
        x_new=x_values;    
    end
    if size(x_new,2)==gpdim
        return x_new
    else
        error("dimensions of x_values do not match those of GP")
    end
end

function scalar(x)
    assert(length(x) == 1)
    return x[1]
end


end
