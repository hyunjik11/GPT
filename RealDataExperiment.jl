@everywhere using GPT_SGLD
@everywhere using Distributions
@everywhere using DataFrames
#using GaussianProcess

if 1==0
tic()
f=SECov(length_scale,1);
gp=GP(0,f,4);
gp_post=GPpost(gp,Xtrain,ytrain,sigma);
gp_pred=Mean(gp_post,Xtest);
toc()
println("RMSE for GP=",ytrainStd*norm(ytest-gp_pred)/sqrt(N-Ntrain))
end

@everywhere data=DataFrames.readtable("Folds5x2_pp.csv", header = true);
@everywhere data = convert(Array,data);
@everywhere N=size(data,1);
@everywhere D=4;
@everywhere Ntrain=5000;
@everywhere seed=17;
@everywhere length_scale=1.4332;
@everywhere sigma=0.2299;
@everywhere Xtrain = data[1:Ntrain,1:D];
@everywhere ytrain = data[1:Ntrain,D+1];
@everywhere XtrainMean=mean(Xtrain,1); 
@everywhere XtrainStd=zeros(1,D);
@everywhere	for i=1:D
	    XtrainStd[1,i]=std(Xtrain[:,i]);
	end
@everywhere ytrainMean=mean(ytrain);
@everywhere ytrainStd=std(ytrain);
@everywhere Xtrain = GPT_SGLD.datawhitening(Xtrain);
@everywhere ytrain=GPT_SGLD.datawhitening(ytrain);
@everywhere Xtest = (data[Ntrain+1:end,1:D]-repmat(XtrainMean,N-Ntrain,1))./repmat(XtrainStd,N-Ntrain,1);
@everywhere ytest = (data[Ntrain+1:end,D+1]-ytrainMean)/ytrainStd;

if 1==0
@everywhere burnin=17;
@everywhere maxepoch=3;
@everywhere t=((80,1e-13),(85,1e-13),(90,1e-13),(95,1e-13),(100,1e-13),(80,1e-14),(85,1e-14),(90,1e-14),(95,1e-14),(100,1e-14),(80,1e-15),(85,1e-15),(90,1e-15),(95,1e-15),(100,1e-15),(80,1e-16),(85,1e-16),(90,1e-16),(95,1e-16),(100,1e-16));
    @parallel for  i=1:length(t)
        m,n=t[i]
	phitrain=GPT_SGLD.feature(Xtrain,n,length_scale,seed);
	phitest=GPT_SGLD.feature(Xtest,n,length_scale,seed);
        for r=10:10:30
            for Q=50:50:200
                for i=10:10:100
                    for j=14:16
	                epsw=i;epsU=10.0^(-j);
	                tic()
	                GPT_SGLD.SDexp(phitrain,phitest,ytrain,ytest,ytrainStd,seed,sigma,length_scale,n,r,Q,m,epsw,epsU,burnin,maxepoch,"StdOutSiris.txt");
	                toc()
                    end
                end
            end
        end
    end
end



