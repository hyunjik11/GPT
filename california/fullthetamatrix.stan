data {
  int<lower=1> N;
  int<lower=1> Ntrain;
  int<lower=1> n;
  matrix[Ntrain,n] phitrainU;
  matrix[Ntrain,n] phitrainV;
  matrix[N-Ntrain,n] phitestU;
  matrix[N-Ntrain,n] phitestV;
  vector[Ntrain] ytrain;
  real<lower=0> sigma;
}
parameters {
  matrix[n,n] theta;
}
transformed parameters{
  vector[Ntrain] trainpred;
  for (i in 1:Ntrain)
	trainpred[i] <- phitrainU[i]*theta*phitrainV[i]';
}
model {
  for (i in 1:n)
	  theta[i] ~ normal(0,1);
  ytrain ~ normal(trainpred,sigma);
}
generated quantities {
  vector[N-Ntrain] testpred;
  for (i in 1:(N-Ntrain))
	testpred[i] <- phitestU[i]*theta*phitestV[i]';
}
