
GPT_infindep = function(X,y,sigma,n,r,sigmaRBF,q,generator,num_iterations,burnin){
  ##precompute features
  D = ncol(X);N=nrow(X);W=c()
  
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
    
    for (d in 1:q) {
      
       tmp = apply(sapply(1:D,function(z) {  (t(U_array[,I[d,z],z]) %*% b[,,z]) }),1,prod) 
      
#       tmp_array = array(,c(n,1,D))
#       for (j in 1:D){
#         tmp_array[,,j] = U_array[,I[d,j],j]
#       }
#       tmp1 = sapply(1:N,tmp_func)
#       tmp = .C("M_func",as.double(tmp_array),as.double(b),as.integer(nrow(X)),as.integer(n),as.integer(1),as.integer(D),as.double(rep(0,n)),as.double(rep(0,n)),as.double(matrix(0,1,D)),
#                as.double(rep(0,1)),as.double(matrix(0,nrow(X),1)))[[11]]      
#       
    
      var_w = 1/(sum(tmp^2)/(sigma^2) + (1/sigma_w)^2 )
      W[d] = rnorm(1) * sqrt(var_w) + var_w * (tmp %*% y)/(sigma^2) 
    }
    

    ##posterior on the Us 
    for (k in 1:D){
      bk = b[,,-k]
      
      column = sort(unique(I[,k]))
      for (l in column) {
        index = which(I[,k]==l);q1 = length(index); q2 = q-q1
        I_l = matrix(I[index,],q1,D); I_l1 = I[-index,]
        
        tmp_array = array(,c(n,q1,D))
        for (d in 1:D){
          tmp_array[,,d] = U_array[,I_l[,d],d]
        }
        tmp_arrayk = tmp_array[,,-k]; 
        
        res1 = .C("M_func",as.double(tmp_arrayk),as.double(bk),as.integer(nrow(X)),as.integer(n),as.integer(q1),as.integer(D-1),as.double(rep(0,n)),
                  as.double(rep(0,n)),as.double(matrix(0,q1,D-1)),as.double(rep(0,q1)),as.double(matrix(0,nrow(X),q1)))[[11]]
        coeff1 = as.numeric(t(W[index]) %*% matrix(res1,q1,nrow(X)))
        
        tmp_array = array(,c(n,q2,D))
        for (d in 1:D){
          tmp_array[,,d] = U_array[,I_l1[,d],d]
        }         
        res2 = .C("M_func",as.double(tmp_array),as.double(b),as.integer(nrow(X)),as.integer(n),as.integer(q2),as.integer(D),as.double(rep(0,n)),
                  as.double(rep(0,n)),as.double(matrix(0,q2,D)),as.double(rep(0,q2)),as.double(matrix(0,nrow(X),q2)))[[11]]
        coeff2 = as.numeric(t(W[-index]) %*% matrix(res2,q2,nrow(X)))
        
        Phi = coeff1 * t(b[,,k])  
        Precision = 1/(sigma^2) * (t(Phi) %*% Phi) + 1/(sigma_u)^2 * diag(n)
        Mu_U = solve(Precision, b[,,k] %*% ((y-coeff2) * coeff1)/ (sigma^2))
        U_array[,l,k] = solve(chol(Precision),rnorm(n)) + Mu_U        
      }
    }
    
    if(m > burnin) {
      W_array[,m-burnin] = W
      V_array[,,,m-burnin] = U_array
    }
    print(m)
  }
  return(list(W_array=W_array,V_array = V_array,I=I))
}