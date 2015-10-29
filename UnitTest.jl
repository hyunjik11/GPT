using TGP
using PyPlot

import DataFrames

data = DataFrames.readtable("Folds5x2_pp.csv", header = true)
data = convert(Array,data)

#for i = 1:size(data,2)
 #   data[:,i] = (data[:,i] - mean(data[:,i]))/std(data[:,i])
#end

Xtrain = data[1:5000,1:4] ; ytrain = reshape(data[1:5000,5],5000,1)
Xtest = data[5001:end,1:4] ; ytest = reshape(data[5001:end,5],size(data,1)-5000,1)
sigma=0.2299; n=10; r=5; sigmaRBF=1.4332; q=100; num_iterations=3; burnin=1; generator = 123

####Parafac GP######
Mu = Parafac(Xtrain,ytrain,sigma,n,sigmaRBF,generator)
Psi = [prod(feature(Xtest[j,:],n,sigmaRBF,generator)[i,:]) for i = 1:n, j = 1:size(Xtest,1)]
yfit = Psi' * Mu
#plot(ytest[1:10])
#plot(yfit[1:10],color="red",linestyle= "--")

#####tensor GP#######


#RMSE as a function of n
RMSE = Float64[]; timer = Float64[];
for n in round(linspace(20,150,5))
    n = convert(Int64,n)
    now = tic()
    tmp = TensorRes(Xtrain,ytrain,sigma,n,r,sigmaRBF,q,generator,num_iterations,burnin,Xtest,ytest)
    RMSE = push!(RMSE,tmp)
    timer = push!(timer,toc())
    println(RMSE)
end
