#using GaussianProcess
using GPT_SGLD
#using Distributions
using HDF5

n=5;r=2;Q=32;D=5;
sigma_RBF=1.0;length_scale=1.0+0.2*randn(D)
N=100;
X=randn(N,D);
y,w,U,I,phi=fhatdraw(X,n,length_scale,sigma_RBF,r,Q);
y1=y+sqrt(0.1)*randn(N);
y2=y+sqrt(0.01)*randn(N);
y3=y+sqrt(0.001)*randn(N);


#cd("/data/chiffchaff/hkim/")
c=h5open(string("TensorSynthData5D",N,"N.h5"),"w") do file
	write(file,"X",X);
	write(file,"w",w);
	write(file,"U",U);
	write(file,"I",I);
	write(file,"phi",phi);
	write(file,"length_scale",length_scale);	
	write(file,"y1",y1);
	write(file,"y2",y2);
	write(file,"y3",y3);
end



#=
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
=#
