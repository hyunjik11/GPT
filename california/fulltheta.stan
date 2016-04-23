data {
  int<lower=1> Ntrain;
  int<lower=1> n;
  matrix[Ntrain,n] phitrain;
  vector[Ntrain] ytrain;
  real<lower=0> sigma;
}
parameters {
  vector[n] theta;
}
model {
  theta ~ normal(0,1);
  ytrain ~ normal(phitrain*theta,sigma);
}

