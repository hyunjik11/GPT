using HDF5
using GaussianProcess

#if 1==0
file="10000SynthData.h5";
Xtrain=h5read(file,"Xtrain");
XtrainMean=h5read(file,"XtrainMean");
XtrainStd=h5read(file,"XtrainStd");
ytrain=h5read(file,"ytrain");
Xtest=h5read(file,"Xtest");
ytest=h5read(file,"ytest");
#end

if 1==1
tic()
f=SECov(1.4,1);
gp=GP(0,f,4);
gp_post=GPpost(gp,Xtrain,ytrain,0.2);
gp_pred=Mean(gp_post,Xtest);
toc()
println(norm(ytest-gp_pred)/sqrt(5000))
end
