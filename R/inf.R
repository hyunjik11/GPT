
########################################################################################################################################
########################################################################################################################################


GPT_inf = function(X,y,sigma,n,r,sigmaRBF,q,generator,num_iterations,burnin){
 
  ##precompute features
  D = ncol(X);N=nrow(X)
  
  b = array(,c(n,nrow(X),D))
  for (i in 1:nrow(X)){ b[,i,] = feature(X[i,],n,sigmaRBF,generator)}
  
   #initialise U's
  sigma_u = sqrt(1/r)
  sigma_w = sqrt(r^D/q) 
  U_array = array(,c(n,r,ncol(X)))
  for (j in 1:ncol(X)){U_array[,,j] = matrix(sigma_u*rnorm(n*r),n,r)}
  ##initialise W
  I = t(sapply(1:q,function(x) {sample(1:r,D,replace=T)}))
  W_array = array(,c(q,num_iterations-burnin))
  V_array = array(,c(n,r,D,num_iterations-burnin))
  
  ###inference
  for (m in 1:num_iterations){
    
    tmp_array = array(,c(n,q,D))
    for (d in 1:D){
      tmp_array[,,d] = U_array[,I[,d],d]
    }
    
    #     U_phi = sapply(1:nrow(X),function(i) {apply(sapply(1:D,function(z) {  t(U_array[,I[,z],z]) %*% b[,i,z] }),1,prod) })
    
    res = .C("M_func",as.double(tmp_array),as.double(b),as.integer(nrow(X)),as.integer(n),as.integer(q),as.integer(D),as.double(rep(0,n)),as.double(rep(0,n)),as.double(matrix(0,q,D)),
             as.double(rep(0,q)),as.double(matrix(0,nrow(X),q)))[[11]]
    U_phi = matrix(res,q,nrow(X))
    
    tmp = (1/sigma_w)^2 * diag(q) + 1/(sigma^2)*(U_phi %*% t(U_phi))
    Mu = solve(tmp,(U_phi %*% y)/(sigma^2))
    W_I = solve(chol(tmp),rnorm(q)) + Mu
    # W_I = as.numeric(rmvnorm(1,mean=Mu,sigma=solve(tmp)))
     
    if(m > burnin) {
      W_array[,m-burnin] = W_I
      V_array[,,,m-burnin] = U_array
    }
    
    ##posterior on the Us 
    for (k in 1:D){
      
      U_phi_j = U_phi / (t(U_array[,I[,k],k]) %*% b[,,k])
      V = as.numeric(W_I) * U_phi_j
      C = matrix(0,r,N) 
      column = unique(sort(I[,k]))
      for (l in column) {
        index = which(I[,k] == l)
        C[l,] = colSums(matrix(V[index,],length(index),N))
      }
      
      Ck = matrix(rep(C,each=n),n*r,N) * matrix(rep(t(b[,,k]),r),ncol =  N , byrow = TRUE )
      tmp = Ck %*% t(Ck)/(sigma^2) + (1/sigma_u)^2 * diag(n*r)
      Mu_U = solve(tmp,(Ck %*% y) / (sigma^2))
      U_array[,,k] = solve(chol(tmp),rnorm(n*r)) + Mu_U

      U_phi = U_phi_j * (t(U_array[,I[,k],k]) %*% b[,,k])
    } 
    

      # print(m)
    y_fit = t(W_I) %*% U_phi 
    print(y_std* sqrt(sum((y_fit-y)^2)/nrow(X)))
  }
  return(list(W_array=W_array,V_array = V_array,I=I))
}
##########################################################################################################################
###########################################################################################################################


GPT_infcol = function(X,y,sigma,n,r,sigmaRBF,q,generator,num_iterations,burnin){
  ##precompute features
  D = ncol(X);N=nrow(X)
  
  b = array(,c(n,nrow(X),D))
  for (i in 1:nrow(X)){ b[,i,] = feature(X[i,],n,sigmaRBF,generator)}
  
  #initialise U's
  sigma_u = sqrt(1/r)
  sigma_w = sqrt(r^D/q) 
  U_array = array(,c(n,r,ncol(X)))
  for (j in 1:ncol(X)){U_array[,,j] = matrix(sigma_u*rnorm(n*r),n,r)}
  ##initialise W
  I = t(sapply(1:q,function(x) {sample(1:r,D,replace=T)}))
  W_array = array(,c(q,num_iterations-burnin))
  V_array = array(,c(n,r,D,num_iterations-burnin))
  
  ###inference
  for (m in 1:num_iterations){
    
    tmp_array = array(,c(n,q,D))
    for (d in 1:D){
      tmp_array[,,d] = U_array[,I[,d],d]
    }   
    
      res = .C("M_func",as.double(tmp_array),as.double(b),as.integer(nrow(X)),as.integer(n),as.integer(q),as.integer(D),as.double(rep(0,n)),as.double(rep(0,n)),as.double(matrix(0,q,D)),
               as.double(rep(0,q)),as.double(matrix(0,nrow(X),q)))[[11]]
      U_phi = matrix(res,q,nrow(X))
#     U_phi = sapply(1:nrow(X),function(i) {apply(sapply(1:D,function(z) {  t(U_array[,I[,z],z]) %*% b[,i,z] }),1,prod) })
    
    tmp = (1/sigma_w)^2 * diag(q) + 1/(sigma^2)*(U_phi %*% t(U_phi))
    Mu = solve(tmp,(U_phi %*% y)/(sigma^2))
    W = solve(chol(tmp),rnorm(q)) + Mu
        
#     if(m > burnin) {
#       W_array[,m-burnin] = W
#       V_array[,,,m-burnin] = U_array
#     }
    ##posterior on the Us 
    for (k in 1:D){
      bk = b[,,-k]
      
      column = sort(unique(I[,k]))
      
      for (l in column) {
        U_phi_j = U_phi / t(U_array[,I[,k],k]) %*% b[,,k]
        V = as.numeric(W) * U_phi_j
        
        index = which(I[,k]==l);q1 = length(index); 
        coeff2 = as.numeric(t(W[-index]) %*% U_phi[-index,])
   
        coeff1 = colSums(matrix(V[index,],q1,N))
        Phi =  t(b[,,k])  * coeff1
        Precision = 1/(sigma^2) * (t(Phi) %*% Phi) + 1/(sigma_u)^2 * diag(n)
                
        Mu_U = solve(Precision, b[,,k] %*% ((y-coeff2) * coeff1)/ (sigma^2))
        U_array[,l,k] = solve(chol(Precision),rnorm(n)) + Mu_U  
        U_phi = U_phi_j * t(U_array[,I[,k],k]) %*% b[,,k]
      }
    }
    
    if(m > burnin) {
      W_array[,m-burnin] = W
      V_array[,,,m-burnin] = U_array
    }
    # print(m)
    y_fit = t(W) %*% U_phi 
    print(paste("m= ",m,", RMSE= ",y_std* sqrt(sum((y_fit-y)^2)/nrow(X))))
  }
  return(list(W_array=W_array,V_array = V_array,I=I))
}

########################################################################################################
######################################################################################################

inf_VI = function(X,y,sigma,n,r,sigmaRBF,Q,generator,num_iterations,burnin){
  ##precompute features
  D = ncol(X); N = nrow(X);ELBO=c();y_fit=matrix(,N,num_iterations);Mu_W=c();Sigma_W =c()
  
  b = array(,c(n,nrow(X),D))
  for (i in 1:nrow(X)){ b[,i,] = feature(X[i,],n,sigmaRBF,generator)}
  
  sigma_u = sqrt(1/r)
  sigma_w = sqrt(r^D/Q) 
  Mu_array = array(,c(n,r,ncol(X)))
  for (d in 1:D) {Mu_array[,,d] = sigma_u * rnorm(n*r)}
  #   for (d in 1:D) {Mu_array[,,d] = rep(0,n*r)}
  Sigma_array = array(,c(n,n,r,D))
  for (d in 1:D) {
    for (l in 1:r) {
      Sigma_array[,,l,d] = sigma_u * diag(n)
    }
  }
  Mu_array_0 = Mu_array
  Sigma_array_0 = Sigma_array
  I = t(sapply(1:Q,function(x) {sample(1:r,D,replace=T)}))
  
  
  for (m in 1:num_iterations){
    ### W ######
    

      func_w = function(i) {
        mat = matrix(0,Q,Q) 
        
        for (k in 1:Q) {
          for (l in k:Q) {
            
            mat[k,l] = prod(sapply(1:D,function(z) {
                      if (I[k,z] != I[l,z]) {
                        out = b[,i,z] %*% Mu_array[,I[k,z],z] * (Mu_array[,I[l,z],z] %*% b[,i,z])
                      }
                      else {
                        tmp = Mu_array[,I[k,z],z] %*% t(Mu_array[,I[l,z],z]) + Sigma_array[,,I[k,z],z] 
                        out = t(b[,i,z]) %*% tmp %*% b[,i,z]
                      }
                    return(out)
                    }))
          }
        }
        mat = mat + t(mat) - Diagonal(Q,diag(mat))
        return(mat)
      }

      Tmp = lapply(1:N,func_w)
      tmp_mat = Reduce("+",Tmp)
      
      tmp_array = array(,c(n,Q,D))
      for (d in 1:D){
        tmp_array[,,d] = Mu_array[,I[,d],d]
      }  
      res = .C("M_func",as.double(tmp_array),as.double(b),as.integer(nrow(X)),as.integer(n),as.integer(Q),as.integer(D),as.double(rep(0,n)),as.double(rep(0,n)),as.double(matrix(0,Q,D)),
               as.double(rep(0,Q)),as.double(matrix(0,nrow(X),Q)))[[11]]
      U_phi = matrix(res,Q,nrow(X))
      
      precision_W =  (1/sigma^2)* tmp_mat +  (1/sigma_w)^2*diag(Q)
      Sigma_W = solve(precision_W)
      Mu_W = solve(precision_W,(U_phi %*% y)/(sigma^2) )
    
      for (k in 1:D) {
        column = sort(unique(I[,k]))
        
        for (l in column) {
          index = which(I[,k]==l);q1 = length(index); index2 = (1:Q)[-index]
          U_phi_j = U_phi / t(Mu_array[,I[,k],k]) %*% b[,,k]
          V = as.numeric(Mu_W) * U_phi_j
          Ea = colSums(matrix(V[index,],q1,N))
          
          Tmp1 = lapply(1:N,function(i) {
             Tmp[[i]][index,index] / as.numeric(t(b[,i,k]) %*% ( Mu_array[,l,k] %*% t(Mu_array[,l,k]) + Sigma_array[,,l,k]) %*% b[,i,k] )
           })
          Ea2 = sqrt(sapply(1:N,function(i) {sum((Mu_W[index] %*% t(Mu_W[index]) + Sigma_W[index,index]) * Tmp1[[i]])}))
  
          
          Eab_func = function(i) {
            Eab_mat = matrix(,q1,Q-q1)
             for (l1 in 1:q1) {
               for (l2 in 1:(Q-q1)) {
                 Eab_mat[l1,l2] = Tmp[[i]][index[l1],index2[l2]]/ as.numeric( t(b[,i,k]) %*% Mu_array[,l,k] )  *
                   (Mu_W[index[l1]] * Mu_W[index2[l2]] + Sigma_W[index[l1],index2[l2]]) 
               }
             }
             return(sum(Eab_mat))
           }
         
          Eab = sapply(1:N,Eab_func)
       
          Phi = Ea2 * t(b[,,k])  
          Precision = 1/(sigma^2) * (t(Phi) %*% Phi) + 1/(sigma_u)^2 * diag(n)
          Mu_array[,l,k] = solve(Precision, (b[,,k] %*% (y*Ea-Eab))/ sigma^2  + (1/sigma_u^2)*as.numeric(Mu_array_0[,l,k])) 
          Sigma_array[,,l,k] = solve(Precision)      
          U_phi = U_phi_j * t(Mu_array[,I[,k],k]) %*% b[,,k]
          
        }
      }     
     print(m)
     y_fit[,m] = as.numeric(t(Mu_W) %*% U_phi)
  }
  return(list(Mu_W,Mu_array,I=I))
}


RMSE = c()
plot(y,type="l",ylim=range(y_fit,y,na.rm=T))
for (i in 1:num_iterations) {
  lines(y_fit[,i],col=i+1)
  RMSE[i] = round(sqrt(sum((y_fit[,i]-y)^2)/length(y)),3)
}



