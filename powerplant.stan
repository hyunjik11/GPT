data {
  int<lower=1> n;
  int<lower=1> randomfeatures;
  int<lower=1> d;
  //matrix[n,d] x;
  //matrix[randomfeatures,d] w; 
  matrix[n,randomfeatures*2] X;
  vector[n] y;

  real<lower=0.00001> sigma;
  real<lower=0> bw;
}
parameters {
  vector[randomfeatures*2] beta;
}

model {
  y ~ normal(X * beta,sigma);
  beta ~ normal(0,1);
  sigma ~ normal(0,1);
  bw ~ normal(.2,.1);
}
