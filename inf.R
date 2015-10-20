
########################################################################################################################################
########################################################################################################################################

GPT_inf = function(X,y,sigma,n,r,sigmaRBF,Q,generator,num_iterations,burnin){ 
  #Gibbs sampling for Tucker GP, sampling each U^(k)
 
  ##precompute features
  D = ncol(X);N=nrow(X)
  
  phi = array(,c(n,N,D)) #features. phi[,i,k]=phi^(k)(x_i)
  for (i in 1:N){ phi[,i,] = feature(X[i,],n,sigmaRBF,generator)}
  
  ##initialise U's
  sigma_u = sqrt(1/r)
  sigma_w = sqrt(r^D/Q) 
  U = array(sigma_u*rnorm(n*r*D),c(n,r,D)) #So U[,,k]=U^(k)
  
  ##Initialise w by sampling the locations of non-zeros in w
  I = t(sapply(1:Q,function(x) {sample(1:r,D,replace=T)})) #Q by D matrix. qth row shows the indices of the qth nonzero element in R
  #could get duplicate rows with prob 1-(r^D-1)(r^D-2)...(r^D-Q+1)/r^(QD)) (birthday paradox). Could pose a small problem
  w_array = array(,c(Q,num_iterations-burnin)) #storing all the w's for each iteration for validation
  U_array = array(,c(n,r,D,num_iterations-burnin)) #ditto for U's
  
  ###inference
  for (m in 1:num_iterations){
    
    tmp_array = array(,c(n,Q,D))
    for (k in 1:D){
      tmp_array[,,k] = U[,I[,k],k] #I[,k] are the indices of the kth dimension of the nonzero values in W
      #So tmp_array[,,k] has the cols of U[,,k] for which we need the inner product with the kth feature vector
    }
    
    # V = sapply(1:nrow(X),function(i) {apply(sapply(1:D,function(z) {  t(U[,I[,z],z]) %*% phi[,i,z] }),1,prod) })
    # Same thing but without sapply:
    
    #V=matrix(,Q,n)
    #for (i in 1:N){
    #  v=rep(1,Q)
    #  for (k in 1:D){
    #    vk=phi[,i,k]%*%tmp_array[,,k]
    #    v=v*vk
    #  }
    #  V[,i]=v;
    #}
    
    res = .C("M_func",as.double(tmp_array),as.double(phi),as.integer(nrow(X)),as.integer(n),as.integer(Q),as.integer(D),as.double(rep(0,n)),as.double(rep(0,n)),as.double(matrix(0,Q,D)),
             as.double(rep(0,Q)),as.double(matrix(0,nrow(X),Q)))[[11]] #implements the above commented R command
    V = matrix(res,Q,nrow(X)) #V is the Q by N matrix, where f(x_i)=sum_over_q{w_q*V_qi}
    
    tmp = (1/sigma_w)^2 * diag(Q) + 1/(sigma^2)*(V %*% t(V)) #precision for posterior of w, which we need to invert to get covariance
    Mu = solve(tmp,(V %*% y)/(sigma^2)) #mu in eqn (9)
    w = solve(chol(tmp),rnorm(Q)) + Mu # w ~ N(Mu,tmp^(-1))
    # w = as.numeric(rmvnorm(1,mean=Mu,sigma=solve(tmp)))
     
    if(m > burnin) { #if passed burn in stage, store the w and U
      w_array[,m-burnin] = w
      U_array[,,,m-burnin] = U
    }
    
    ##posterior on the Us 
    for (k in 1:D){
      U_phi_k = V / (t(U[,I[,k],k]) %*% phi[,,k]) #denom=<I_qk'th col of U^(k),phi^(k)(x_i)>
      #So U_phi_k is a Q x N matrix where U_phi_k[q,i] is expression inside the big brackets in eqn(10)
      B = as.numeric(w) * U_phi_k #Q x N matrix where B[q,i] is summand in formula for a^(k)_l(x_i)
      #why need as.numeric?
      a = matrix(0,r,N) 
      column = unique(sort(I[,k])) #unique elems of set {kth index of the nonzeros of W} - don't think we need sort
      for (l in column) {
        index = which(I[,k] == l)
        a[l,] = colSums(matrix(B[index,],length(index),N)) #colSums(B[index,]) is equivalent, 
        #but written in this way for exception handling when length(index)=1
        #a[l,i]=a^(k)_l(x_i)
      }
      
      Psi_k = matrix(rep(C,each=n),n*r,N) * matrix(rep(t(phi[,,k]),r),ncol =  N , byrow = TRUE ) #Psi_k=Psi^(k)
      Precision = Psi_k %*% t(Psi_k)/(sigma^2) + (1/sigma_u)^2 * diag(n*r) #Precision for posterior of u^(k) i.e. inv(Sigma^(k))
      mu_k = solve(Precision,(Psi_k %*% y) / (sigma^2)) #mu^(l)_k
      U[,,k] = solve(chol(Precision),rnorm(n*r)) + mu_k #sample U^(k) as in eqn(11)

      V = U_phi_k * (t(U[,I[,k],k]) %*% phi[,,k]) #update V
    } 
    

      # print(m)
    y_fit = t(w) %*% V 
    print(y_std* sqrt(sum((y_fit-y)^2)/N))
  }
  return(list(w_array=w_array,U_array = U_array,I=I))
}
##########################################################################################################################
###########################################################################################################################


GPT_infcol = function(X,y,sigma,n,r,sigmaRBF,Q,generator,num_iterations,burnin){ 
  #independent cols Tucker
  
  ##precompute features
  D = ncol(X);N=nrow(X)
  
  phi = array(,c(n,N,D))
  for (i in 1:N){ phi[,i,] = feature(X[i,],n,sigmaRBF,generator)}
  
  #initialise U's
  sigma_u = sqrt(1/r)
  sigma_w = sqrt(r^D/Q) 
  U = array(,c(n,r,D))
  for (j in 1:ncol(X)){U[,,j] = matrix(sigma_u*rnorm(n*r),n,r)}
  ##initialise w
  I = t(sapply(1:Q,function(x) {sample(1:r,D,replace=T)}))
  w_array = array(,c(Q,num_iterations-burnin))
  U_array = array(,c(n,r,D,num_iterations-burnin))
  
  ###inference
  for (m in 1:num_iterations){
    
    tmp_array = array(,c(n,Q,D))
    for (d in 1:D){
      tmp_array[,,d] = U[,I[,d],d]
    }   
    
      res = .C("M_func",as.double(tmp_array),as.double(phi),as.integer(nrow(X)),as.integer(n),as.integer(Q),as.integer(D),as.double(rep(0,n)),as.double(rep(0,n)),as.double(matrix(0,Q,D)),
               as.double(rep(0,Q)),as.double(matrix(0,nrow(X),Q)))[[11]]
      V = matrix(res,Q,nrow(X))
#     V = sapply(1:nrow(X),function(i) {apply(sapply(1:D,function(z) {  t(U[,I[,z],z]) %*% b[,i,z] }),1,prod) })
    
    tmp = (1/sigma_w)^2 * diag(Q) + 1/(sigma^2)*(V %*% t(V))
    Mu = solve(tmp,(V %*% y)/(sigma^2))
    w = solve(chol(tmp),rnorm(Q)) + Mu
        
    ##posterior on the Us 
    for (k in 1:D){
      column = sort(unique(I[,k]))
      
      for (l in column) {
        U_phi_k = V / t(U[,I[,k],k]) %*% phi[,,k]
        B = as.numeric(w) * U_phi_k 
        
        index = which(I[,k]==l) #I_l  
        q1 = length(index); 
        coeff2 = as.numeric(t(w[-index]) %*% V[-index,]) #vector length N - second term in eqn(12)
   
        coeff1 = colSums(matrix(B[index,],q1,N)) #vector length N where coeff1[i]=a^(k)_l(x_i)
        Phi =  t(phi[,,k])  * coeff1 # N x n matrix Phi^(k,l)
        Precision = 1/(sigma^2) * (t(Phi) %*% Phi) + 1/(sigma_u)^2 * diag(n) #Precision for posterior of lth col of U^(k) in eqn(13)
                
        mu_kl = solve(Precision, phi[,,k] %*% ((y-coeff2) * coeff1)/ (sigma^2)) #mu^(k)_l in eqn(13)
        U[,l,k] = solve(chol(Precision),rnorm(n)) + mu_kl  #lth col of U^(k) sampled from N(mu_k,Precision^(-1))
        V = U_phi_k * t(U[,I[,k],k]) %*% phi[,,k] #update V
      }
    }
    
    if(m > burnin) {
      w_array[,m-burnin] = w
      U_array[,,,m-burnin] = U
    }
    # print(m)
    y_fit = t(w) %*% V 
    print(paste("m= ",m,", RMSE= ",y_std* sqrt(sum((y_fit-y)^2)/N)))
  }
  return(list(w_array=w_array,U_array = U_array,I=I))
}

########################################################################################################
######################################################################################################

inf_VI = function(X,y,sigma,n,r,sigmaRBF,Q,generator,num_iterations,burnin){ #variational inference
  ##precompute features
  D = ncol(X); N = nrow(X);ELBO=c();y_fit=matrix(,N,num_iterations);Mu_W=c();Sigma_W =c()
  
  b = array(,c(n,nrow(X),D))
  for (i in 1:nrow(X)){ b[,i,] = feature(X[i,],n,sigmaRBF,generator)}
  
  sigma_u = sqrt(1/r)
  sigma_w = sqrt(r^D/Q) 
  MU = array(,c(n,r,ncol(X)))
  for (d in 1:D) {MU[,,d] = sigma_u * rnorm(n*r)}
  #   for (d in 1:D) {MU[,,d] = rep(0,n*r)}
  Sigma_array = array(,c(n,n,r,D))
  for (d in 1:D) {
    for (l in 1:r) {
      Sigma_array[,,l,d] = sigma_u * diag(n)
    }
  }
  MU_0 = MU
  Sigma_array_0 = Sigma_array
  I = t(sapply(1:Q,function(x) {sample(1:r,D,replace=T)}))
  
  
  for (m in 1:num_iterations){ 
    ### W ######
    

      func_w = function(i) { 
        mat = matrix(0,Q,Q) 
        
        for (k in 1:Q) { #these nested for loops look like it's gonna be slow
          for (l in k:Q) {
            
            mat[k,l] = prod(sapply(1:D,function(z) {
                      if (I[k,z] != I[l,z]) {
                        out = b[,i,z] %*% MU[,I[k,z],z] * (MU[,I[l,z],z] %*% b[,i,z])
                      }
                      else {
                        tmp = MU[,I[k,z],z] %*% t(MU[,I[l,z],z]) + Sigma_array[,,I[k,z],z] 
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
        tmp_array[,,d] = MU[,I[,d],d]
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
          U_phi_k = U_phi / t(MU[,I[,k],k]) %*% b[,,k]
          V = as.numeric(Mu_W) * U_phi_k
          Ea = colSums(matrix(V[index,],q1,N))
          
          Tmp1 = lapply(1:N,function(i) {
             Tmp[[i]][index,index] / as.numeric(t(b[,i,k]) %*% ( MU[,l,k] %*% t(MU[,l,k]) + Sigma_array[,,l,k]) %*% b[,i,k] )
           })
          Ea2 = sqrt(sapply(1:N,function(i) {sum((Mu_W[index] %*% t(Mu_W[index]) + Sigma_W[index,index]) * Tmp1[[i]])}))
  
          
          Eab_func = function(i) {
            Eab_mat = matrix(,q1,Q-q1)
             for (l1 in 1:q1) {
               for (l2 in 1:(Q-q1)) {
                 Eab_mat[l1,l2] = Tmp[[i]][index[l1],index2[l2]]/ as.numeric( t(b[,i,k]) %*% MU[,l,k] )  *
                   (Mu_W[index[l1]] * Mu_W[index2[l2]] + Sigma_W[index[l1],index2[l2]]) 
               }
             }
             return(sum(Eab_mat))
           }
         
          Eab = sapply(1:N,Eab_func)
       
          Phi = Ea2 * t(b[,,k])  
          Precision = 1/(sigma^2) * (t(Phi) %*% Phi) + 1/(sigma_u)^2 * diag(n)
          MU[,l,k] = solve(Precision, (b[,,k] %*% (y*Ea-Eab))/ sigma^2  + (1/sigma_u^2)*as.numeric(MU_0[,l,k])) 
          Sigma_array[,,l,k] = solve(Precision)      
          U_phi = U_phi_k * t(MU[,I[,k],k]) %*% b[,,k]
          
        }
      }     
     print(m)
     y_fit[,m] = as.numeric(t(Mu_W) %*% U_phi)
  }
  return(list(Mu_W,MU,I=I))
}




