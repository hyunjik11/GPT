data {
  int<lower=1> N;
  int<lower=1> Ntrain;
  int<lower=1> n1;
  int<lower=1> n2;
  matrix[n1,n1] phiU; //transpose of cholesky factor. ie. phiU*phiU'=Ku
  matrix[n2,n2] phiV; //transpose of cholesky factor. ie. phiV*phiV'=Kv
  int indtrainU[Ntrain];
  int indtrainV[Ntrain];
  int indtestU[N-Ntrain];
  int indtestV[N-Ntrain];
  vector[Ntrain] ytrain;
  real<lower=0> sigma;
}
parameters {
  matrix[n1,n2] theta;
}
transformed parameters{
  matrix[n1,n2] pred;
  vector[Ntrain] trainpred;
  pred <- phiU*theta*phiV';
  for (i in 1:Ntrain)
	  trainpred[i] <- pred[indtrainU[i],indtrainV[i]];
}
model {
  for (i in 1:n1)  
	  theta[i] ~ normal(0,1);
  ytrain ~ normal(trainpred,sigma);
}
generated quantities {
  vector[N-Ntrain] testpred;
  for (i in 1:(N-Ntrain))
	testpred[i] <- pred[indtestU[i],indtestV[i]];
}
