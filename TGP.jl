module TGP

import Distributions
export feature, Parafac, GPT_inf, datawhitening, TensorRes

function feature(x,n,sigmaRBF,generator)
    D = length(x)
    srand(generator)
    Z = randn(n,D)/sigmaRBF
    b = rand(n,D)
    x = repmat(x,n,1)
    phi = sqrt(2/n)*cos(x .* Z + b*2*pi)
    return phi
end

function datawhitening(x) 
    for i = 1:size(x,2)   
        x[:,i] = (x[:,i] - mean(x[:,i]))/std(x[:,i])   
    end
    return(x)
end


function Parafac(X,y,sigma,n,sigmaRBF,generator)
    X = datawhitening(X);
    y = datawhitening(y); 
    N,D = size(X)
    sigma_w = sqrt(n^(D-1))
    Psi = [prod(feature(X[j,:],n,sigmaRBF,generator)[i,:]) for i = 1:n, j = 1:N]
    ##posterior
    invSigma = float(1/(sigma^2) * Psi * Psi' + (1/sigma_w^2)*eye(n))
    Mu = \(invSigma,float(1/(sigma^2) *(Psi * y)))
  return Mu
end

 
function GPT_inf(X,y,sigma,n,r,sigmaRBF,q,generator,num_iterations,burnin)
   
    X = datawhitening(X);
    y = datawhitening(y); 
  
    N,D = size(X)

    W_array = Array(Float64,q,num_iterations-burnin)
    V_array = Array(Float64,n,r,D,num_iterations-burnin)
    b = Array(Float64,n,D,N)
    for i in 1:N  b[:,:,i] = feature(X[i,:],n,sigmaRBF,generator)  end


     #initialise U's
    sigma_u = sqrt(1/r)
    sigma_w = sqrt(r^D/q) 
    U_array = [sigma_u*randn(n,r) for j in 1:D]
    I = rand(Distributions.DiscreteUniform(1, r),q,D)

     for m in 1:num_iterations
        
        
        V = [prod([(U_array[d][:,I[i,d]]' * b[:,d,j])[1] for d = 1:D ]) for i = 1:q, j = 1:N]

        invSigma_w = 1/(sigma^2) * V * V' + (1/sigma_w^2)*eye(q)
        Mu_w = \(invSigma_w,1/(sigma^2) *(V * y))
        W_I = \(chol(invSigma_w,:U),randn(q)) + Mu_w

        if m > burnin
          W_array[:,m-burnin] = W_I
          [V_array[:,:,j,m-burnin] = U_array[j] for j = 1:D]
        end

        for k in 1:D
          V_k = float([V[l,m]/(U_array[k][:,I[l,k]]' * b[:,k,m])[1] for l=1:q, m=1:N])
          V_kk = W_I .* V_k
          C = Array(Float64,r,N)
          for l in unique(I[:,k])
              C[l,:] = collect(sum(V_kk[findin(I[:,k],l),:],1))
          end
            Ck = repeat(C,inner=[n,1],outer=[1,1]).*repmat(squeeze(b[:,k,:],2),r,1)   
          invSigma_U = Ck * Ck'/(sigma^2) + (1/sigma_u)^2 * eye(n*r)
          Mu_U = \(invSigma_U, (Ck * y) / (sigma^2))    
          U_array[k]= reshape(\(factorize(invSigma_U),randn(n*r)) + Mu_U,n,r)
          V =[V_k[l,m] *  (U_array[k][:,I[l,k]]' *  b[:,k,m])[1] for l=1:q, m=1:N]
        end
    end

    return W_array, V_array, I
end


function TensorRes(Xtrain,ytrain,sigma,n,r,sigmaRBF,q,generator,num_iterations,burnin,X,y)
    W,U,I = GPT_inf(Xtrain,ytrain,sigma,n,r,sigmaRBF,q,generator,num_iterations,burnin)
    ystd = std(y)
    X = datawhitening(X);
    y = datawhitening(y); 
    
    N,D = size(X)
    yfit = Array(Float64,size(X,1),size(W,2))    
    b = Array(Float64,n,D,N)
    for i in 1:N  b[:,:,i] = feature(X[i,:],n,sigmaRBF,generator)  end

    for m in 1:size(W,2)
        U_array = U[:,:,:,m]
        V = [prod([(U_array[:,I[i,d],d]' * b[:,d,j])[1] for d = 1:D ]) for i = 1:q, j = 1:N]
        yfit[:,m] = W[:,m]' * V
    end
    yfit = mean(yfit,2)
    RMSE = ystd*sqrt(sum((yfit - y).^2)/N)
    return RMSE
end
end
