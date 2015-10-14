
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
  I = t(sapply(1:Q,function(x) {sample(1:r,D,replace=T)}))
  
  
  for (m in 1:num_iterations){
    ### W ######
    
    for (q in 1:Q) {    
    
    func1 = function(i) {
      prod(sapply(1:D,function(z) { t(b[,i,z]) %*% (Sigma_array[,,I[q,z],z] +  Mu_array[,I[q,z],z] %*% t( Mu_array[,I[q,z],z])) %*% b[,i,z] })) 
    }
    
    tmp = sum(sapply(1:N,func1))
    Sigma_W[q] = 1/((1/sigma^2)*tmp + (1/sigma_w)^2 )
    tmp2 = apply(sapply(1:D,function(z) {  (t(Mu_array[,I[q,z],z]) %*% b[,,z]) }),1,prod) 
    Mu_W[q] = Sigma_W[q] * (tmp2 %*% y)/(sigma^2) 
    }
    
    
    for (k in 1:D) {
      bk = b[,,-k]
      
      column = sort(unique(I[,k]))
      for (l in column) {
        index = which(I[,k]==l);q1 = length(index); q2 = q-q1
        I_l = matrix(I[index,],q1,D); I_l1 = I[-index,]
        
        tmp_array = array(,c(n,q1,D))
        for (d in 1:D){
          tmp_array[,,d] = Mu_array[,I_l[,d],d]
        }
        tmp_arrayk = tmp_array[,,-k]; 
        
        
#         res1 = sapply(1:nrow(X),function(i) {apply(matrix(sapply((1:D)[-k],function(z) {  
#           t(Mu_array[,I_l[,z],z]) %*% b[,i,z] }),q1,(D-1)),1,prod) })
#         
        
        res1 = .C("M_func",as.double(tmp_arrayk),as.double(bk),as.integer(nrow(X)),as.integer(n),as.integer(q1),as.integer(D-1),as.double(rep(0,n)),
                  as.double(rep(0,n)),as.double(matrix(0,q1,D-1)),as.double(rep(0,q1)),as.double(matrix(0,nrow(X),q1)))[[11]]
        Ea = as.numeric(t(Mu_W[index]) %*% matrix(res1,q1,nrow(X)))
        
        tmp_array = array(,c(n,q2,D))
        for (d in 1:D){
          tmp_array[,,d] = Mu_array[,I_l1[,d],d]
        }         
#         res2 = .C("M_func",as.double(tmp_array),as.double(b),as.integer(nrow(X)),as.integer(n),as.integer(q2),as.integer(D),as.double(rep(0,n)),
#                   as.double(rep(0,n)),as.double(matrix(0,q2,D)),as.double(rep(0,q2)),as.double(matrix(0,nrow(X),q2)))[[11]]
#        
#         res2 = sapply(1:nrow(X),function(i) {apply(matrix(sapply((1:D),function(z) {  
#         t(Mu_array[,I_l1[,z],z]) %*% b[,i,z] }),q2,D),1,prod) })

        # Eb = as.numeric(t(Mu_W[-index]) %*% matrix(res2,q2,nrow(X)))
        
        
        
        func_a2 = function(i) {
          mat = matrix(0,q1,q1) 
            for (l1 in 1:q1) {
                 for (l2 in l1:q1) {
                   W_tmp = Mu_W[index[l1]] * Mu_W[index[l2]] 
                   if (l1==l2) {W_tmp = W_tmp + Sigma_W[index[l1]]}
                   mat[l1,l2] = W_tmp * prod(sapply((1:D)[-k],function(z) {          
                    cov_tmp = Mu_array[,I_l[l1,z],z] %*% t( Mu_array[,I_l[l2,z],z])
                    if (I_l[l1,z] == I_l[l2,z] ) {cov_tmp = cov_tmp + Sigma_array[,,I_l[l1,z],z] }
                    return ( t(b[,i,z]) %*% cov_tmp %*% b[,i,z] )}))
                    
                 }
            }
          mat = mat + t(mat) - Diagonal(q1,diag(mat))
          return(sum(mat))
        }
        Ea2 = sqrt(sapply(1:N,func_a2))
        
        func_ab = function(i) {
          mat = matrix(0,q1,Q-q1)
          for (l1 in 1:q1) {
            for (l2 in 1:(Q-q1)) {
              W_tmp = Mu_W[index[l1]] * Mu_W[-index][l2] 
              mat[l1,l2] = W_tmp * prod(sapply((1:D)[-k],function(z) {          
                cov_tmp = Mu_array[,I_l[l1,z],z] %*% t( Mu_array[,I_l1[l2,z],z])
                if (I_l[l1,z] == I_l1[l2,z] ) {cov_tmp = cov_tmp + Sigma_array[,,I_l[l1,z],z] }
                return ( t(b[,i,z]) %*% cov_tmp %*% b[,i,z]) })) * (t(Mu_array[,I_l1[l2,k],k])) %*% b[,i,z]    
              
            }
          }
          return(sum(mat))
        }
        Eab = sapply(1:N,func_ab)
   
        Phi = Ea2 * t(b[,,k])  
        Precision = 1/(sigma^2) * (t(Phi) %*% Phi) + 1/(sigma_u)^2 * diag(n)
        Mu_array[,l,k] = solve(Precision, b[,,k] %*% (y*Ea-Eab)  + (1/sigma_u^2)*as.numeric(Mu_array_0[,l,k])/ (sigma^2))
        Sigma_array[,,l,k] = solve(Precision)       
      }
      
    }     

#     Sigma_array2 = array(,c(n*r,n*r,D)) 
#     for (z in 1:D) {
#       Sigma_array2[,,z] = Reduce(adiag,lapply(1:r, function(l) {Sigma_array[,,l,z]}))
#     }
# 
#     EW2_func = function(i) {
#       EW2 = matrix(0,Q,Q); 
#       for (d in 1:Q) {
#         for (k in d:Q) {
#           EW2[d,k] = prod(sapply((1:D),function(z) {
#             t(b[,i,z]) %*% (Sigma_array2[,,z] [((I[d,z]-1)*n + 1):(I[d,z]*n),((I[k,z]-1)*n + 1):(I[k,z]*n)] 
#                             + Mu_array[,I[d,z],z] %*% t(Mu_array[,I[k,z],z]))   %*% b[,i,z] } ))  
#         }
#       }
#       return( EW2 + t(EW2) - Diagonal(q,diag(EW2)))
#     }
# 
# #     
#     tmp_W = Reduce("+",lapply(1:N,EW2_func))
# 
#     WW = matrix(,Q,Q);
#     for (d in 1:Q) {
#       for (k in 1:Q) {
#         if (d==k) {WW[d,k] = Sigma_W[d] + Mu_W[d] * Mu_W[k]}
#         else {WW[d,k] = Mu_W[d] * Mu_W[k] }
#       }
#     }
#     Ey2 = sum( WW * tmp_W);
# 
# 
#     Ey = Mu_W %*% tmp2
#     ELBO[m] = as.numeric( -1/(2*sigma^2) * Ey2 + (1/sigma^2) *  Ey %*% y  -1/(2*sigma_w^2) * sum(Sigma_W + Mu_W^2) -                    
#           1/(2*sigma_u^2) * sum(sapply(1:D, function(z) {sapply(1:r. function(v) {trace(Sigma_array[,,v,z]) + t(Mu_array[,v,z]) %*% Mu_array[,v,z] } )})))


#     
    print(m)

    y_fit[,m] = sapply(1:nrow(X),function(i) { matrix(Mu_W,1,r^D) %*% Reduce(kronecker,lapply(rev(1:D),function(z) {  t(Mu_array[,,z]) %*% b[,i,z] })) })
  }
  return(list(Mu_W,Mu_array,I=I))
}


RMSE = c()
plot(y,type="l",ylim=range(y_fit,y,na.rm=T))
for (i in 1:num_iterations) {
  lines(y_fit[,i],col=i+1)
  RMSE[i] = round(sqrt(sum((y_fit[,i]-y)^2)/length(y)),3)
}




