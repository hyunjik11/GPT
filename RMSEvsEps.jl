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
    ax[:set_zlim](4,10)
    ax[:set_xlabel]("epsw")
    ax[:set_ylabel]("log epsU")
end


@everywhere x=70:5:100; 
@everywhere y=15:17;
@everywhere xgrid=repmat(x',3,1);
@everywhere ygrid=repmat(y,1,7);
if 1==0
z=[5.61,5.609,5.609,5.609,9.02,9.015,9.015,9.015,5.07,5.07,5.07,5.07,8.09,8.08,8.08,8.08,4.86,4.86,4.86,4.86, 7.62,7.61,7.61,7.61,4.73,4.73,4.73,4.73,7.33,7.33,7.33,7.33,4.65,4.65,4.65,4.65,7.13,7.13,7.13,7.13];
plot_3d(xgrid,ygrid,z);
end
