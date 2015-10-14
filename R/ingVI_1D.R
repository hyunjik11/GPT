
inf_VI = function(X,y,sigma,n,r,sigmaRBF,q,generator,num_iterations,burnin){
  ##precompute features
  D = ncol(X); N = nrow(X);ELBO=c();y_fit=matrix(,N,num_iterations)
  
  b = matrix(,n,N)
  for (i in 1:nrow(X)){ b[,i] = feature(X[i,],n,sigmaRBF,generator)}
  
  sigma_u = sqrt(1/r)
  sigma_w = sqrt(r^D/q) 
  Mu_W_0 = sigma_w*rnorm(q)
  Sigma_W_0 = sigma_w*diag(q)
  Mu_W = Mu_W_0; Sigma_W = Sigma_W_0  
  ##initialise non-zero entries of W
#   I = sapply(1:q,function(x) {sample(1:r,D,replace=T)})
  I = 1:q
  
  for (m in 1:num_iterations){
    ### U ######
    
    Ea = Mu_W
    EA = matrix(0,r,r)
    
    for (r1 in 1:q) {
      for (r2 in 1:q) {
        EA[I[r1],I[r2]] = Sigma_W[r1,r2] + Mu_W[r1] * Mu_W[r2]
      }
    }
    
    EA_func = function(i) {
      return (kronecker(EA, b[,i] %*% t(b[,i])))
    }
    
    
    Precision_j = Reduce("+",lapply(1:N,EA_func))/(sigma^2) + (1/sigma_u)^2 *diag(n*r)
    Sigma_U = matrix(solve(Precision_j),n*r,n*r)
    
    func1 = function(i) { matrix(kronecker(Ea,b[,i]),r*n,1)}     
    Mu_U = matrix(Sigma_U %*% sapply(1:N,func1) %*% y / sigma^2,n,r)
    
#     F = sapply(1:nrow(X),function(i) {t(Mu_U[,I]) %*% b[,i] })
    F = t(Mu_U) %*% b
    
    EW2_func = function(i) {
      EW2 = matrix(0,q,q); 
      for (d in 1:q) {
        for (k in d:q) {
          EW2[d,k] =
            t(b[,i]) %*% (Sigma_U[((I[d]-1)*n + 1):(I[d]*n),((I[k]-1)*n + 1):(I[k]*n)] 
                            + Mu_U[,I[d]] %*% t(Mu_U[,I[k]]))   %*% b[,i] 
        }
      }
      return( EW2 + t(EW2) - Diagonal(q,diag(EW2)))
    }
    
    tmp_W = Reduce("+",lapply(1:N,EW2_func))
    Sigma_W = solve ((1/sigma^2) * tmp_W + 1/(sigma_w^2)*diag(q) )
    Mu_W = (1/sigma^2) * Sigma_W %*% (F %*% y + (1/sigma_w^2)*Mu_W_0)
    
    W = rep(0,r)
    W[I] = as.numeric(Mu_W)
   
    ##ELBO
    WW = matrix(,q,q);
    for (d in 1:q) {
      for (k in 1:q) {
        WW[d,k] = Sigma_W[d,k] + Mu_W[d] * Mu_W[k]
      }
    }
  
    Ey2 = sum( WW * tmp_W);
    
    
    ELBO[m] = as.numeric( -1/(2*sigma^2) * Ey2 + 
                            (1/sigma^2) *  sapply(1:nrow(X),function(i) { matrix(W,1,r) %*% t(Mu_U) %*% b[,i] }) %*% y + 
                            q/2 + t(Mu_W) %*% solve(Sigma_W) %*% Mu_W + 1/2 *sum(diag(Sigma_W)) +
                              n*r/2 + matrix(Mu_U,1,n*r) %*% solve(Sigma_U) %*% matrix(Mu_U,n*r,1) + 0.5*sum(diag((Sigma_U)))
                              )
    
    print(m)
    y_fit[,m] = matrix(t(b) %*% Mu_U[,I] %*% Mu_W, N,1)
  }
  
}

# RMSE = c()
# plot(y,type="l",ylim=range(y_fit,y,na.rm=T))
# for (i in 1:num_iterations) {
#   lines(y_fit[,i],col=i+1)
#   RMSE[i] = round(sqrt(sum((y_fit[,i]-y)^2)/length(y)),3)   #### need to rescale the data in practice
# }

