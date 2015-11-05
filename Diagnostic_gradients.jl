using GPT_SGLD
using HDF5

file="10000SynthData.h5";
Xtrain=h5read(file,"Xtrain");
XtrainMean=h5read(file,"XtrainMean");
XtrainStd=h5read(file,"XtrainStd");
ytrain=h5read(file,"ytrain");
Xtest=h5read(file,"Xtest");
ytest=h5read(file,"ytest");

sigma=0.2;sigmaRBF=1.4;n=100;r=10;Q=100;m=100;seed=17;
D=size(Xtrain,2);
phitrain=feature(Xtrain,n,sigmaRBF,seed);
L=sample(0:(r^D-1),Q,replace=false)
I=Array(Int32,Q,D)
for q in 1:Q
    I[q,:]=digits(L[q],r,D)+1
end

function logp(phitrain,ytrain,w,U,I,sigma) #log p(y|x,w,U)
    n,D,train_size=size(phitrain)
    temp=Array(Float64,D,r,train_size)
    for i=1:train_size
        for l=1:r
            for k=1:D
		temp[k,l,i]=dot(phitrain[:,k,i],U[:,l,k])
	    end
        end
    end

    # compute V st V[q,i]=prod_{k=1 to D}(temp[k,I[q,k],i])
    V=ones(Q,train_size)
    for i=1:train_size
        for q=1:Q
	    for k=1:D
		V[q,i]*=temp[k,I[q,k],i];
	    end
        end
    end

    # compute fhat where fhat[i]=V[:,i]'w
    fhat=Array(Float64,train_size)
    for i=1:train_size
	fhat[i]=dot(V[:,i],w)
    end

    return -(norm(ytrain-fhat))^2/(2*sigma^2)
end

function dlogpdw(phitrain,ytrain,w,U,I,sigma)
    n,D,train_size=size(phitrain)
    temp=Array(Float64,D,r,train_size)
    for i=1:train_size
        for l=1:r
            for k=1:D
		temp[k,l,i]=dot(phitrain[:,k,i],U[:,l,k])
	    end
        end
    end

    # compute V st V[q,i]=prod_{k=1 to D}(temp[k,I[q,k],i])
    V=ones(Q,train_size)
    for i=1:train_size
        for q=1:Q
	    for k=1:D
		V[q,i]*=temp[k,I[q,k],i];
	    end
        end
    end

    # compute fhat where fhat[i]=V[:,i]'w
    fhat=Array(Float64,train_size)
    for i=1:train_size
	fhat[i]=dot(V[:,i],w)
    end

    return V*(y_batch-fhat)/(sigma^2)
end

function dlogpdU(phitrain,ytrain,w,U,I,sigma)


