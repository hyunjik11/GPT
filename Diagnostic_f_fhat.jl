using GaussianProcess
using GPT_SGLD
using PyPlot
using Distributions

function createmesh(interval_start,interval_end,npts)
    x=linspace(interval_start,interval_end,npts)
    y=linspace(interval_start,interval_end,npts)
    xgrid=repmat(x',npts,1)
    ygrid=repmat(y,1,npts)
    grid=Array(Float64,npts^2,2); k=1;
    for i=1:npts
	for j=1:npts
		grid[k,:]=[x[i] y[j]];
		k+=1;
	end
    end
    return xgrid,ygrid,grid
end

function plot_3d(xgrid,ygrid,y)
    npts=size(xgrid,1)
    z=reshape(y,npts,npts)
    fig=figure()
    ax=fig[:add_subplot](1,1,1, projection = "3d")
    ax[:plot_surface](xgrid,ygrid,z,rstride=2, edgecolors="k", cstride=2, cmap=ColorMap("coolwarm"), alpha=0.8, linewidth=0.25)
end

function fhatdraw(X,n,sigmaRBF,r,Q)
    N,D=size(X)
    phi=feature(X,n,sigmaRBF,rand(1:2^10))
    sigma_w=sqrt(n^D/Q)
    w=sigma_w*randn(Q)
    U=Array(Float64,n,r,D)
    for k=1:D
        Z=randn(r,n)
        U[:,:,k]=transpose(\(sqrtm(Z*Z'),Z)) #sample uniformly from V_{n,r}
    end
    
    l=sample(0:(r^D-1),Q,replace=false)
    I=Array(Int32,Q,D)
    for q in 1:Q
        I[q,:]=digits(l[q],r,D)+1
    end
    return pred(w,U,I,phi)
end

l=1.4;
f=SECov(l,1);
mygp=GP(0,f,2);

xgrid,ygrid,X=createmesh(-1,1,50);
z=GPrand(mygp,X);
fhat=fhatdraw(X,100,l,10,50);
plot_3d(xgrid,ygrid,z);
plot_3d(xgrid,ygrid,fhat);
    

    
