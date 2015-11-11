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
    nptsx=size(xgrid,2)
    nptsy=size(ygrid,1)
    z=reshape(y,nptsy,nptsx)
    fig=figure()
    ax=fig[:add_subplot](1,1,1, projection = "3d")
    ax[:plot_surface](xgrid,ygrid,z,rstride=2, edgecolors="k", cstride=2, cmap=ColorMap("coolwarm"), alpha=0.8, linewidth=0.25)
    ax[:set_zlim](4,7)
    ax[:set_xlabel]("epsw")
    ax[:set_ylabel]("log epsU")
end

x=5:5:50; y=10:15;
xgrid=repmat(x',6,1);
ygrid=repmat(y,1,10);
z=[5.9178,6,5.9516384,6.15157,6.6392,6.70178,5.79181,6.07061,5.4848,5.14313,6.5247,5.026,4.91, 5.80899,4.7831,5.27,4.7102,4.519,4.973,6.0282,5.131366,5.115,4.9718,6.7198,6.397,5.3027,4.695973, 5.23279,5.7737,5.212,4.7367,5,4.9373,5.25574,5.1488,5.41406,5.27494,6.0485,4.9010,5.13471,5.18966, 5.5,5.85776,17.817,4.990,4.934,5.52,5.2347,6.09877,4.6114,6.5829,4.77172,5.2339,4.8821,4.9270, 4.916,4.7076,4.5257, 4.4237,5];
plot_3d(xgrid,ygrid,z);
