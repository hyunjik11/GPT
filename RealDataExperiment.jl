using GPT_SGLD
using Distributions
using DataFrames
using GaussianProcess

if 1==0
tic()
f=SECov(sigmaRBF,1);
gp=GP(0,f,4);
gp_post=GPpost(gp,Xtrain,ytrain,sigma);
gp_pred=Mean(gp_post,Xtest);
toc()
println("RMSE for GP=",ytrainStd*norm(ytest-gp_pred)/sqrt(N-Ntrain))
end

if 1==1
@everywhere data=readtable("Folds5x2_pp.csv", header = true);
@everywhere data = convert(Array,data);
@everywhere N=size(data,1);
@everywhere D=4;
@everywhere Ntrain=5000;
@everywhere seed=17;
@everywhere sigmaRBF=1.4332;
@everywhere sigma=0.2299;
@everywhere Xtrain = data[1:Ntrain,1:D];
@everywhere ytrain = data[1:Ntrain,D+1];
@everywhere XtrainMean=mean(Xtrain,1); 
@everywhere XtrainStd=Array(Float64,1,D);
	for i=1:D
	    XtrainStd[1,i]=std(Xtrain[:,i]);
	end
@everywhere ytrainMean=mean(ytrain);
@everywhere ytrainStd=std(ytrain);
@everywhere Xtrain = datawhitening(Xtrain);
@everywhere ytrain=datawhitening(ytrain);
@everywhere Xtest = (data[Ntrain+1:end,1:D]-repmat(XtrainMean,N-Ntrain,1))./repmat(XtrainStd,N-Ntrain,1);
@everywhere ytest = (data[Ntrain+1:end,D+1]-ytrainMean)/ytrainStd;
@everywhere burnin=17;
@everywhere maxepoch=3;
@everywhere t=((50,50),(50,100),(50,150),(50,200),(100,50),(100,100),(100,150),(100,200),(200,50),(200,100),(200,150),(200,200),(400,50),(400,100),(400,150),(400,200))
    @parallel for  i=1:length(t)
        m,n=t[i]
	phitrain=feature(Xtrain,n,sigmaRBF,seed);
	phitest=feature(Xtest,n,sigmaRBF,seed);
        for r=10:10:30
            for Q=50:50:200
                for i=10:10:100
                    for j=14:16
	                epsw=i;epsU=10.0^(-j);
	                tic()
	                SDexp(phitrain,phitest,ytrain,ytest,ytrainStd,seed,sigma,sigmaRBF,n,r,Q,m,epsw,epsU,burnin,maxepoch);
	                toc()
                    end
                end
            end
        end
    end

end




