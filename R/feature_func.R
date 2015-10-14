library(plyr)

feature = function(x,n,sigmaRBF,generator){
  D = length(x)
  set.seed(generator)
  Z = matrix(rnorm(n*D)/sigmaRBF,D,n)
  b = matrix(runif(n*D),D,n)
  phi = t(sqrt(2/n)*cos(x * Z + b*2*pi))
  return(phi)
}
#previous bug
# feature = function(x,n,sigmaRBF,generator){
#   D = length(x)
#   set.seed(generator)
#   Z = matrix(rnorm(n*D)/sigmaRBF,n,D)
#   b = matrix(runif(n*D),n,D)
#   phi = sqrt(2/n)*cos(Z * x + b*2*pi)
#   return(phi)
# }