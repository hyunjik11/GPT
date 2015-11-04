using GaussianProcess
#using GPT_SGLD
using PyPlot

sigmaRBF=1.4; sigma=0.2;
f=SECov(sigmaRBF,1);
mygp=GP(0,f,2);
x=linspace(-1,1,20);
y=linspace(-1,1,20);
xgrid=repmat(x',20,1);
ygrid=repmat(y,1,20);
X=Array(Float64,20*20,2); k=1;
for i=1:20
	for j=1:20
		X[k,:]=[x[i] y[j]];
		k+=1
	end
end
function gpdraw(gp,X)
	z=GPrand(gp,X);
	z=reshape(z,20,20)
	fig=figure()
	ax=fig[:add_subplot](1,1,1, projection = "3d") 
	ax[:plot_surface](xgrid,ygrid,z,rstride=2, edgecolors="k", cstride=2, cmap=ColorMap("gray"), alpha=0.8, linewidth=0.25)
end

gpdraw(mygp,grid)

	
