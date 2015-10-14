
GPdraw = function(X,sigmaRBF,seed){
  kernelRBF = function(x,y)  exp(-1/(2*sigmaRBF^2)*(t(x-y)%*%(x-y)))
  nr = nrow(X)
  cov = matrix(0,nr,nr)
  for (i in 1:nr){
    for (j in i:nr){
      cov[i,j] = kernelRBF(X[i,],X[j,])
    }
  }
  cov = cov + t(cov) - diag(nr)
  set.seed(seed)
  return(as.vector(mvrnorm(1,rep(0,nr),cov)))
}



GPdrawsvd = function(X,n,r,sigmaRBF,D,generator){
  
  feature = function(x){
    phi = matrix(,n,D)

    for (j in 1:D){
      set.seed(generator+j)
      Z = rnorm(n)/sigmaRBF
      b = runif(n)
      tmp = Z*x[j]
      phi[,j] = sqrt(2/n)*cos(tmp + b*2*pi)
    }
    return(kronecker(phi[,1],phi[,2]))
  }

  fea_mat = apply(X,1,feature)
  W = matrix(n/r*rnorm(r^2),r,r)
  U = rbing.Op(diag(n), diag(n))
  V = rbing.Op(diag(n), diag(n))
  U = U[,1:r];V=V[,1:r]
  Theta = matrix(U %*% W %*% t(V),n^2,1)
  f =  t(t(Theta) %*% fea_mat)
  return(f)
}


####################################################################################



