
inf_VI = function(X,y,sigma,n,r,sigmaRBF,q,generator,num_iterations,burnin){
  ##precompute features
  D = ncol(X); N = nrow(X);ELBO=c();y_fit=matrix(,N,num_iterations)
  
  b = array(,c(n,nrow(X),D))
  for (i in 1:nrow(X)){ b[,i,] = feature(X[i,],n,sigmaRBF,generator)}
  
  sigma_u = sqrt(1/r)
  sigma_w = sqrt(r^D/q) 
  Mu_array = array(,c(n,r,ncol(X)))
  for (d in 1:D) {Mu_array[,,d] = sigma_u * rnorm(n*r)}
  Sigma_array = array(,c(n*r,n*r,D))
  for (d in 1:D) {Sigma_array[,,d] = sigma_u * diag(n*r)}
  Mu_array_0 = Mu_array
  I = t(sapply(1:q,function(x) {sample(1:r,D,replace=T)}))
  
  
  for (m in 1:num_iterations){
    ### W ######
    
    F = sapply(1:nrow(X),function(i) {apply(sapply(1:D,function(z) {  t(Mu_array[,I[,z],z]) %*% b[,i,z] }),1,prod) })
    
    EW2_func = function(i) {
      EW2 = matrix(0,q,q); 
      for (d in 1:q) {
        for (k in d:q) {
          EW2[d,k] = prod(sapply((1:D),function(z) {
            t(b[,i,z]) %*% (Sigma_array[,,z] [((I[d,z]-1)*n + 1):(I[d,z]*n),((I[k,z]-1)*n + 1):(I[k,z]*n)] 
                            + Mu_array[,I[d,z],z] %*% t(Mu_array[,I[k,z],z]))   %*% b[,i,z] } ))  
        }
      }
      return( EW2 + t(EW2) - Diagonal(q,diag(EW2)))
    }
    
    
    tmp_W = Reduce("+",lapply(1:N,EW2_func))
    Sigma_W = solve ((1/sigma^2) * tmp_W + 1/(sigma_w^2)*diag(q) )
    Mu_W = (1/sigma^2) * Sigma_W %*% (F %*% y)  
    W = array(0,rep(r,D))
    W[I] = as.numeric(Mu_W)
    
    for (j in 1:D){
      
      Wj = matrix(0,r,r^(D-1))
      index_func = function(ind) { sum(sapply(1:(D-1),function(z) { (ind[z]-1)*r^(z-1)})) +1 }
      index = cbind(I[,j],apply(I[,-j],1,index_func))
      
      Wj = matrix(0,r,q)
      for (l in 1:q) {  Wj[index[,1][l], l] = W[I][l]   }
      
      Wj = k_unfold(as.tensor(W),j)@data
      
      Ea = matrix(,r,nrow(X));
      for (i in 1:nrow(X)){ 
        Ea[,i] = Wj %*% Reduce(kronecker,lapply(rev(1:D)[-j],function(z){t(Mu_array[,,z]) %*% b[,i,z]}))
      }
  
      EU2_func = function(i) {
        EU2 = matrix(0,r,r); 
        
        EW2j = matrix(0,q,q); 
        for (d in 1:q) {
          for (k in d:q) {
            EW2j[d,k] = prod(sapply((1:D)[-j],function(z) {
              t(b[,i,z]) %*% (Sigma_array[,,z] [((I[d,z]-1)*n + 1):(I[d,z]*n),((I[k,z]-1)*n + 1):(I[k,z]*n)] 
                              + Mu_array[,I[d,z],z] %*% t(Mu_array[,I[k,z],z]))   %*% b[,i,z] } ))  
          }
        }
        EW2j =  EW2j + t(EW2j) - Diagonal(q,diag(EW2j))
        
        for (r1 in 1:r) {
          for (r2 in 1:r) {
            
            W_tmp = matrix(0,q,q);
            for (d in 1:q) {
              for (k in 1:q) {
                if (I[d,j] == r1 & I[k,j] == r2) {
                  W_tmp[d,k] = Sigma_W[d,k] + Mu_W[d] * Mu_W[k]         
                }
              }
            }
            
            EU2[r1,r2] = sum(W_tmp * EW2j)
          }
        }
        return(kronecker(EU2, b[,i,j] %*% t(b[,i,j])))
      }
      
      
      Precision_j = Reduce("+",lapply(1:N,EU2_func))/(sigma^2) + (1/sigma_u)^2 *diag(n*r) 
      Sigma_array[,,j] = matrix(solve(Precision_j),n*r,n*r)
      
      func1 = function(i) { matrix(kronecker(Ea,b[,i,j]),r*n,1)}  
      Mu_array[,,j] = Sigma_array[,,j] %*% (sapply(1:N,func1) %*% y + (1/sigma_u^2)*as.numeric(Mu_array_0[,,j])) / sigma^2
      
    }
    
    ##ELBO
    WW = matrix(,q,q);
    for (d in 1:q) {
      for (k in 1:q) {
        WW[d,k] = Sigma_W[d,k] + Mu_W[d] * Mu_W[k]
      }
    }
    Ey2 = sum( WW * tmp_W);
    
    
    ELBO[m] = as.numeric( -1/(2*sigma^2) * Ey2 + 
                            (1/sigma^2) *  sapply(1:nrow(X),function(i) { matrix(W,1,r^D) %*% Reduce(kronecker,lapply(rev(1:D),function(z) {  t(Mu_array[,,z]) %*% b[,i,z] })) }) %*% y + 
                            q/2 + t(Mu_W) %*% solve(Sigma_W) %*% Mu_W + 1/2 *sum(diag(Sigma_W)) +
                            sum(sapply(1:D,function(z) {
                              n*r/2 + matrix(Mu_array[,,z],1,n*r) %*% solve(Sigma_array[,,z]) %*% matrix(Mu_array[,,z],n*r,1) +0.5*sum(diag((Sigma_array[,,z])))
                            }))  )
    
    print(m)
         y_fit[,m] = sapply(1:nrow(X),function(i) { matrix(W,1,r^D) %*% Reduce(kronecker,lapply(rev(1:D),function(z) {  t(Mu_array[,,z]) %*% b[,i,z] })) })
  }
  return(list(Mu_W,Mu_array,I=I))
}

# 
# RMSE = c()
# plot(y,type="l",ylim=range(y_fit,y,na.rm=T))
# for (i in 1:num_iterations) {
#   lines(y_fit[,i],col=i+1)
#   RMSE[i] = round(y_std *sqrt(sum((y_fit[,i]-y)^2)/length(y)),3)
# }
# plot(RMSE,type="l",xlab="number of iterations",ylab="RMSE",
#      main="RMSE on training set, N=10, D=3, n=r=10,q=50")
# plot(ELBO,main="ELBO, VI",xlab="number of iterations",type="l")
# 
# Mu_W = res$Mu_W; Mu_array = res$Mu_array
# y_test_fit = t(Mu_W) %*% sapply(1:nrow(X_test),function(i) {apply(sapply(1:D,function(z) {  t(Mu_array[,I[,z],z]) %*% b[,i,z] }),1,prod) })
