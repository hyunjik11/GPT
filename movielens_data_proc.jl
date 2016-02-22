@everywhere using DataFrames
@everywhere using HDF5

#process ratings data only for ml 100lk
@everywhere Rating = readdlm("/homes/hkim/GPT/ml-100k/u.data",Float64);

@everywhere Ntrain = 80000;
@everywhere Ntest = 20000;
@everywhere ytrain = Rating[1:Ntrain,3];
@everywhere ytest = Rating[Ntrain+1:Ntrain+Ntest,3];
c=h5open("ml100k.h5","w") do file
	write(file,"train_triples",Rating[1:Ntrain,:]);
	write(file,"test_triples",Rating[Ntrain+1:Ntrain+Ntest,:]);
end
