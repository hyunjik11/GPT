
theta_inf = function(X,y,sigma,n,sigmaRBF,generator){
  ##precompute features
  D = ncol(X)
  Phi = apply(X,1,function(x){Reduce(kronecker,alply(array(feature(x,n,sigmaRBF,generator),c(n,D)),2)) })

    ##posterior
  Sigma = solve(1/(sigma^2) * Phi %*% t(Phi) + diag(n^D))
  theta = 1/(sigma^2) * Sigma %*% (Phi %*% y)
  return(theta)
}

parafac_inf = function(X,y,sigma,n,sigmaRBF,generator,diagonal){
  ##precompute features
  D = ncol(X);N=nrow(X);
  
  sigma_w = sqrt(n^(D-1));
  Psi = sapply(1:nrow(X_train),function(i) {apply(feature(X_train[i,],n,sigmaRBF,generator),1,prod) })
  
  ##posterior
  tmp = 1/(sigma^2) * Psi %*% t(Psi) + (1/sigma_w^2) * diag(n)
  Mu = solve(tmp, 1/(sigma^2) *(Psi %*% y))
  return(Mu)
}



# 
# parafac_inf = function(X,y,sigma,n,sigmaRBF,q,generator,diagonal){
#   ##precompute features
#   D = ncol(X);N=nrow(X);
#   
#   if (diagonal == T) {
#     q = n;
#     I = matrix( rep((1:q),D), q,D)
#   }
#   else {I = t(sapply(1:q,function(x) {sample(1:n,D,replace=T)}))}
#   
#   sigma_w = sqrt(n^D/q);
#   b = array(,c(n,nrow(X),D))
#   for (i in 1:nrow(X)){ b[,i,] = feature(X[i,],n,sigmaRBF,generator)}
#    
#   Psi = sapply(1:nrow(X),function(i) {apply(sapply(1:D,function(z) {b[I[,z],i,z]}),1,prod) })
# 
#   ##posterior
#   tmp = 1/(sigma^2) * Psi %*% t(Psi) + (1/sigma_w^2) * diag(q)
#   Mu = solve(tmp, 1/(sigma^2) *(Psi %*% y))
#   return(list(W = Mu,I=I))
# }







W_inf = function(X,y,sigma,n,r,sigmaRBF,q,num_iterations,burnin){
  D = ncol(X);  res = matrix(,num_iterations+1,n^D)
  Phi = apply(X,1,function(x){Reduce(kronecker,alply(array(feature(x,n,sigmaRBF,generator),c(n,D)),2)) })
  Phi_vec_array = array(,c(n^(D-1),nrow(X),D))   ###Phi_{-j}
  for (j in 1:D) {
    Phi_vec_array[,,j] = apply(X,1,function(x){Reduce(kronecker,alply(array(feature(x,n,sigmaRBF,generator)[,-j],c(n,D-1)),2)) })
  }
  b = array(,c(n,nrow(X),D))
  for (j in 1:D){
    for (i in 1:nrow(X)){
      b[,i,j] = feature(X[i,],n,sigmaRBF,generator)[,j]
    }
  }
  ##precompute features
  sigma_u = sqrt(1/r)   #1
  sigma_w = 1 #sqrt(r^D) #sqrt(r^D/q) #sqrt(1/q)
  U_array = array(,c(n,r,ncol(X)))
  for (j in 1:ncol(X)){U_array[,,j] = matrix(sigma_u*rnorm(n*r),n,r)}
  ##initialise W
  w = rep(0,r^D);   
  I = sort(sample(1:(r^D))[1:q])  ; 

  for (m in 1:num_iterations){
  
    U_phi = apply(X,1,function(x){
      Reduce(kronecker,lapply(1:D,function(z){t(U_array[,,z]) %*% feature(x,n,sigmaRBF,generator)[,z]}))[I] } )   
    Sigma = chol2inv(chol( (1/sigma_w)^2 * diag(q) + (1/sigma^2)*(U_phi %*% t(U_phi)) ))
    Mu = Sigma %*% (U_phi %*% y)/(sigma^2) 
    w[I] = mvrnorm(1,Mu,Sigma)
    W = array(w,rep(r,D))

    if (m ==1 ) {res[1,] = Reduce(kronecker,alply(U_array,3))[,I] %*% w[I]}
    
    ##posterior on the Us 
    for (j in 1:D){
      
      Wj = k_unfold(as.tensor(W),D-j+1)@data
      
      #form U_(minus j)
      if (D==2) {Uj = U_array[,,(1:D)[-j]]}
      else {Uj = Reduce(kronecker,alply(array(U_array[,,-j],c(n,r,D-1)),3))}
      
      a = matrix(,r,nrow(X));C = matrix(,r*n,nrow(X));
      for (i in 1:nrow(X)){ a[,i] = Wj %*% (t(Uj) %*% Phi_vec_array[,i,j]) 
                            C[,i] = kronecker(a[,i],b[,i,j])
      }
      
      Sigma_U = solve(C %*% t(C)/(sigma^2) + (1/sigma_u)^2 * diag(n*r))
      Mu_U = Sigma_U %*% (C %*% y) / (sigma^2)
      U_array[,,j] = matrix(mvrnorm(1,Mu_U,Sigma_U),n,r)
      
    }

    TMP = Reduce(kronecker,alply(U_array,3))[,I] %*% w[I]
    if ( m > burnin) {res[m+1,] = (TMP + res[m,])/2 }
    else { res[m+1,] = TMP }  
    print(m)
  }
  return(res)
}
  







