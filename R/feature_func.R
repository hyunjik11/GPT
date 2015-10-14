library(plyr)

feature = function(x,n,sigmaRBF,generator){
  D = length(x)
  set.seed(generator)
  Z = matrix(rnorm(n*D)/sigmaRBF,D,n)
  b = matrix(runif(n*D),D,n)
  phi = t(sqrt(2/n)*cos(x * Z + b*2*pi))
  return(phi)
}

fea_notensor = function(x,m,sigmaRBF,generator){
  set.seed(generator)
  D = length(x)
  w = matrix(rnorm(m*D)/sigmaRBF,m,D)
  b = runif(m)
  return(sqrt(2/m)*cos(w %*% x + b*2*pi))
}

# feature = function(x,n,sigmaRBF,generator){
#   D = length(x)
#   set.seed(generator)
#   Z = matrix(rnorm(n*D)/sigmaRBF,n,D)
#   b = matrix(runif(n*D),n,D)
#   phi = sqrt(2/n)*cos(Z * matrix(sample(rep(as.numeric(x),n)),n,D) + b*2*pi)
#   return(phi)
# }

# feature = function(x,n,sigmaRBF,generator){
#   D = length(x)
#   set.seed(generator)
#   Z = matrix(rnorm(n*D)/sigmaRBF,n,D)
#   b = matrix(runif(n*D),n,D)
#   phi = sqrt(2/n)*cos(Z * x + b*2*pi)
#   return(phi)
# }