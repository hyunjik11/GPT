using GaussianProcess
using GPT_SGLD
using Distributions
using HDF5

sigmaRBF=1.4; sigma=0.2;
f=SECov(sigmaRBF,1)
gp=GP(0,f,4)
N=10000; D=4; Ntrain=5000;
x1=rand(Uniform(0,50),N);
x2=rand(Uniform(30,100),N);
x3=rand(Uniform(1000,1020),N);
x4=rand(Uniform(50,100),N);
X=[x1 x2 x3 x4];
Xtrain = X[1:Ntrain,1:D];
XtrainMean=mean(Xtrain,1); 
XtrainStd=Array(Float64,1,D);
for i=1:D
    XtrainStd[1,i]=std(Xtrain[:,i]);
end
Xtrain = datawhitening(Xtrain);
Xtest = (X[Ntrain+1:end,1:D]-repmat(XtrainMean,N-Ntrain,1))./repmat(XtrainStd,N-Ntrain,1);
X=[Xtrain;Xtest];
y=GPrand(gp,X)+sigma*randn(N);
ytrain = y[1:Ntrain];
ytest = y[Ntrain+1:end];
c=h5open("SynthData1000.h5","w") do file
	write(file,"Xtrain",Xtrain);
	write(file,"XtrainMean",XtrainMean);
	write(file,"XtrainStd",XtrainStd);
	write(file,"ytrain",ytrain);
	write(file,"Xtest",Xtest);
	write(file,"ytest",ytest);
end
