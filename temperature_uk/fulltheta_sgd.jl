@everywhere using HDF5
#@everywhere file="temp_final_chol.h5";
@everywhere file="temp_final_eig.h5";
@everywhere phiU=h5read(file,"phiU"); 
@everywhere phiV=h5read(file,"phiV");
@everywhere indtrainU=h5read(file,"indtrainU"); 
@everywhere indtrainV=h5read(file,"indtrainV");
@everywhere indtestU=h5read(file,"indtestU"); 
@everywhere indtestV=h5read(file,"indtestV");
@everywhere ytrain=h5read(file,"ytrain"); 
@everywhere ytest=h5read(file,"ytest");
@everywhere sigma=0.2611;
@everywhere s=4.4960;
@everywhere Ntrain=length(indtrainV); 
@everywhere Ntest=length(indtestV);

@everywhere function fulltheta_sgd(phiU::Array,phiV::Array,indtrainU::Array,indtrainV::Array,indtestU::Array,indtestV::Array,
ytrain::Vector,ytest::Vector,sigma::Real,s::Real,epsilon::Real,m::Integer,maxepoch::Integer)
	n1=size(phiU,1); n2=size(phiV,1);
	Ntrain=length(indtrainV); Ntest=length(indtestV);
	theta=randn(n1,n2);
	numbatches=numbatches=int(ceil(Ntrain/m))
	trainRMSE=Array(Float64,maxepoch)
	testRMSE=Array(Float64,maxepoch)
	for epoch=1:maxepoch
		perm=randperm(Ntrain)
		for batch=1:numbatches
			idx=(m*(batch-1)+1):min(m*batch,Ntrain)
			idx=perm[idx]
			batch_size=length(idx)
			grad_theta=zeros(n1,n2);
			for i in idx
				phi_u=phiU[indtrainU[i],:];
				phi_v=phiV[indtrainV[i],:];
				grad_theta+=(ytrain[i]-phi_u*theta*phi_v').*phi_u'*phi_v
			end
			grad_theta*=Ntrain/(batch_size*sigma^2)
			theta+=epsilon*(grad_theta-theta)			
		end
		trainpred=Array(Float64,Ntrain)
		for i=1:Ntrain
			trainpred[i]=(phiU[indtrainU[i],:]*theta*phiV[indtrainV[i],:]')[1]
		end
		testpred=Array(Float64,Ntest)
		for i=1:Ntest
			testpred[i]=(phiU[indtestU[i],:]*theta*phiV[indtestV[i],:]')[1]
		end
		trainRMSE[epoch]=sqrt(mean((ytrain-trainpred).^2))*s
		testRMSE[epoch]=sqrt(mean((ytest-testpred).^2))*s
		println("epoch=$epoch,trainRMSE=",trainRMSE[epoch],",testRMSE=",testRMSE[epoch])
	end
	return trainRMSE,testRMSE
end

@everywhere maxepoch=100
@everywhere m=100
@everywhere epsilon=1e-4
	println("epsilon=$epsilon")
	trainRMSE,testRMSE=fulltheta_sgd(phiU,phiV,indtrainU,indtrainV,indtestU,indtestV,
ytrain,ytest,sigma,s,epsilon,m,maxepoch)

#plot(trainRMSE,label="trainRMSE");
#plot(testRMSE,label="testRMSE");
#title("RMSE for fulltheta model on UK temperature data using SGD");
#legend(); 
	
	
	
	
