using GPT_SGLD
using HDF5
using Distributions

function logp(phitrain,ytrain,w,U,I,sigma) #log p(y|x,w,U)
    n,D,train_size=size(phitrain)
    Q=length(w)
    r=size(U,2)

    # compute <phi^(k)(x_i),U^(k)_{.l}> for all k,l,batch and store in temp
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

function dlogpdwU(phitrain,ytrain,w,U,I,sigma)
    n,D,train_size=size(phitrain)
    Q=length(w)
    r=size(U,2)
    
    # compute <phi^(k)(x_i),U^(k)_{.l}> for all k,l,batch and store in temp
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

    gradw=V*(ytrain-fhat)/(sigma^2);
    
    U_phi=Array(Float64,Q,train_size,D)
    for k=1:D
	for i=1:train_size
	    for q=1:Q
		U_phi[q,i,k]=V[q,i]/temp[k,I[q,k],i]
	    end
	end
    end
    # now compute a_l^(k)(x_i) for l=1,...,r k=1,..,D and store in A
    A=zeros(r,D,train_size)
    for i=1:train_size
        for k=1:D
            for l in unique(I[:,k])
                index=findin(I[:,k],l) #I_l
                A[l,k,i]=dot(U_phi[index,i,k],w[index])
            end
        end
    end
    # compute Psi as in (12)
    Psi=Array(Float64,n*r,train_size,D)
    for i=1:train_size
        for k=1:D
            Psi[:,i,k]=kron(A[:,k,i],phitrain[:,k,i])
        end
    end
    # can now compute gradU of log p(y|x,w,U)
    gradU=Array(Float64,n,r,D)
    for k=1:D
        gradU[:,:,k]=reshape(Psi[:,:,k]*(ytrain-fhat)/(sigma^2),n,r)
    end

    return gradw,gradU
end

if 1==1
file="10000SynthData.h5";
Xtrain=h5read(file,"Xtrain");
XtrainMean=h5read(file,"XtrainMean");
XtrainStd=h5read(file,"XtrainStd");
ytrain=h5read(file,"ytrain");
Xtest=h5read(file,"Xtest");
ytest=h5read(file,"ytest");

sigma=0.2;sigmaRBF=1.4;n=10;r=5;Q=10;m=100;seed=17;
D=size(Xtrain,2);
sigma_w=sqrt(n^D/Q);
phitrain=feature(Xtrain,n,sigmaRBF,seed);
L=sample(0:(r^D-1),Q,replace=false);
I=Array(Int32,Q,D);
for q in 1:Q
    I[q,:]=digits(L[q],r,D)+1;
end
U=Array(Float64,n,r,D)
for k=1:D
    Z=randn(r,n);
    U[:,:,k]=transpose(\(sqrtm(Z*Z'),Z)); #sample uniformly from a*V_{n,r}
end
w=sigma_w*randn(Q);

loglikelihood=logp(phitrain,ytrain,w,U,I,sigma);

epsw=1e-6;
fdw=Array(Float64,Q);
for q=1:Q
    tempw=deepcopy(w); 
    tempw[q]+=epsw;
    fdw[q]=(logp(phitrain,ytrain,tempw,U,I,sigma)-loglikelihood)/epsw;
end
fdU=Array(Float64,n,r,D);
epsU=1e-6;
for k=1:D
    for l=1:r
	for i=1:n
	    tempU=deepcopy(U);
	    tempU[i,l,k]+=epsU
	    fdU[i,l,k]=(logp(phitrain,ytrain,w,tempU,I,sigma)-loglikelihood)/epsU
	end
    end
end
gradw,gradU=dlogpdwU(phitrain,ytrain,w,U,I,sigma);
println("meangradw=",mean(gradw)," stdgradw=",std(gradw))
println("std_diffw=",std(gradw-fdw))
for k=1:D
    println("k=",k)
    println("meangradUk=",mean(gradU[:,:,k])," stdgradUk=",std(gradU[:,:,k]))
    println("std_diffUk=",std(gradU[:,:,k]-fdU[:,:,k]))
end
end





