# 
# inf_VI = function(X,y,sigma,n,r,sigmaRBF,q,generator,num_iterations,burnin){
#  
#   D = ncol(X);
#   
#   b = array(,c(n,nrow(X),D))
#   for (j in 1:D){
#     for (i in 1:nrow(X)){
#       b[,i,j] = feature(X[i,],n,sigmaRBF,generator)[,j]
#     }
#   }
#   
#   sigma_u = sqrt(1/r)
#   sigma_w = sqrt(r^D/q) 
#   Mu_array = array(,c(n,r,ncol(X)))
#   for (j in 1:ncol(X)){Mu_array[,,j] = matrix(sigma_u*rnorm(n*r),n,r)}
#   Sigma_array = array(,c(n*r,n*r,D))
#   for (j in 1:ncol(X)){Sigma_array[,,j] = sigma_u*diag(n*r)}
# 
#   I = sort(sample(1:(r^D))[1:q])
#   W = array(w,rep(r,D))
#   
#   
#   for (m in 1:num_iterations){
#     
#     EU_phi2_func = function(i) {
#     res = 1
#       
#     for (j in rev(1:D)) {  
#       EU_phi2 = matrix(,r,r) 
#       
#       for (r1 in 1:r) {
#         for (r2 in 1:r) {
#           
#           EU_phi2[r1,r2] = t(b[,i,j]) %*% (Sigma_array[((r1-1)*n+1) : (r1*n) ,((r2-1)*n+1) : (r2*n),j] 
#           - Mu_array[,r1,j] %*% t(Mu_array[,r1,j])) %*% b[,i,j]
#          
#           }
#         }
#         res = kronecker(res,EU_phi2)
#       }
#     return(res)
#     }
#      
#     U_phi = apply(X,1,function(x){
#       Reduce(kronecker,lapply(1:D,function(z){t(Mu_array[,,z]) %*% feature(x,n,sigmaRBF,generator)[,z]}))} )   
#     
#     Precision_W = Reduce("+",lapply(1:N,EU_phi2_func))/(sigma^2) + 1/(sigma_w)^2*diag(r^D)
#     Sigma_W = solve(Precision_W)
#     Mu_W = Sigma_W %*% (U_phi %*% y)/(sigma^2) 
#     W = array(Mu_W,rep(r,D))
#      
#     for (j in 1:D){
#       
#       Wj = k_unfold(as.tensor(W),D-j+1)@data
#       
#       Ea = matrix(,r,nrow(X));
#       for (i in 1:nrow(X)){ 
#         Ea[,i] = Wj %*% Reduce(kronecker,lapply(rev((1:D)[-j]),function(z){t(Mu_array[,,z]) %*% b[,i,z]}))
#       }
#             
#       
#       EA_func = function(i) {
#         EA = matrix(0,r,r);
#         
#         for (d in 1:r) {
#           for (k in d:r) {
#             
#             coeff_U = matrix(,r^D,r^D); coeff_W = matrix(,r^D,r^D)
#             for (l1 in 1:r^D) {
#               for (l2 in 1:r^D) {           
#                 coeff_U[l1,l2] = prod(sapply((1:D)[-j],function(z) {
#                   t(b[,i,z]) %*% (Sigma_array[,,z] [((I[index_d[l1],j]-1)*n + 1):(I[index_d[l1],j]*n),((I[index_k[l2],j]-1)*n + 1):(I[index_k[l2],j]*n)] 
#                                   - Mu_array[,I[index_d[l1],j],z] %*% t(Mu_array[,I[index_k[l2],j],z]))   %*% b[,i,z] } ))
#                 
#                 coeff_W[l1,l2] = Sigma_W[index_d[l1],index_k[l2]] - Mu_W[index_d[l1]] * Mu_W[index_k[l2]]
#                 
#               }
#             }         
#             EA[d,k] = sum(coeff_U * coeff_W)
#             }
#           }
#         }
#         EA = EA + t(EA) - Diagonal(r,diag(EA))
#         return(kronecker(EA, b[,i,j] %*% t(b[,i,j])))
#       }
#       
#       Precision_j = Reduce("+",sapply(1:N,EA_func))/(sigma^2) + (1/sigma_u)^2 *diag(n*r)
#       
#       Sigma_array[,,j] = matrix(solve(Precision_j),n*r,n*r)
#       
#       func1 = function(i) { kronecker(Ea[,i],b[,i,j]) }
#       Mu_array[,,j] = matrix(Sigma_array[,,j] %*% (sapply(1:N,func1) %*% y) / sigma^2,n,r)
#     }
#     
#     ##ELBO
#     
#     
#     WW = matrix(,q,q);
#     for (d in 1:q) {
#       for (k in 1:q) {
#         WW[d,k] = Sigma_W[d,k] - Mu_W[d] * Mu_W[k]
#       }
#     }
#     Ey2 = sum( WW * tmp_W);
#     
#     
#     ELBO[m] = as.numeric( -N/(2*sigma^2) * Ey2 + (1/sigma^2) *  t(Mu_W) %*% U_phi %*% y + 
#                             q/2 - t(Mu_W) %*% solve(Sigma_W) %*% Mu_W + 1/2 *sum(diag(Sigma_W)) +
#                             sum(sapply(1:D,function(z) {
#                               n*r/2 + matrix(Mu_array[,,z],1,n*r) %*% solve(Sigma_array[,,z]) %*% matrix(Mu_array[,,z],n*r,1)
#                               -0.5*sum(diag((Sigma_array[,,z])))
#                             }))  )
#     
#     print(m)
#     for (i in 1:length(y)) {
#       y_fit[i,m] = Ea[,i] %*% Mu_array[,,j] %*% b[,i,j]
#     }
#   }
#   
# }
# 
# 
# RMSE = c()
# plot(y,type="l",ylim=range(y_fit,y,na.rm=T))
# for (i in 1:num_iterations) {
#   lines(y_fit[,i],col=i+1)
#   RMSE[i] = round(sqrt(sum((y_fit[,i]-y)^2)/length(y)),3)
# }