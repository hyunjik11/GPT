\name{GPT_inf}
\Rdversion{1.0}
\alias{gpOut}
\title{Gibbs sampling inference for tucker GP.}
\description{
  learning W and U of the tucker GP.
}
\usage{
GPT_inf = function(X,y,sigma,n,r,sigmaRBF,q,generator,num_iterations,burnin)
}
\arguments{
  \item{X}{training inputs.}
  \item{y}{training outputs.}
  \item{sigma} {Gaussian noise.}
  \item{n} {number of features in each dimension.}
  \item{r} {dimension of lower rank approximation.}
  \item{sigmaRBF} {Gaussian RBF kernel length scale.}
  \item{q} {number of non-zero entries in w.}
  \item{generator} {seed for generating random features.}
  \item{num_iterations} {number of Gibbs iterations.}
  \item{burnin} {number of burn in period.}
}
\value{
  \item{W_array}{Path of W.}
  \item{V_array} {Path of U's.}
  \item{I} {Index matrix of non-zero entries in W.}
}

