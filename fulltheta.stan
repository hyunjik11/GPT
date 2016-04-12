data {
  int<lower=1> N;
  int<lower=1> Ntrain;
  int<lower=1> n;
  matrix[Ntrain,n] phitrain;
  matrix[N-Ntrain,n] phitest;
  vector[Ntrain] ytrain;
  vector[N-Ntrain] ytest;
  real<lower=0> sigma;
  real<lower=0> ytrainStd;
}
parameters {
  vector[n] theta;
}
model {
  theta ~ normal(0,1);
  ytrain ~ normal(phitrain*theta,sigma);
}
generated quantities {
  vector[Ntrain] trainpred;
  vector[N-Ntrain] testpred;
  real trainRMSE;
  real testRMSE;
  trainpred <- phitrain * theta;
  testpred <- phitest * theta;
  trainRMSE <- ytrainStd*sqrt((ytrain-trainpred)'*(ytrain-trainpred)/Ntrain);
  testRMSE <- ytrainStd*sqrt((ytest-testpred)'*(ytest-testpred)/(N-Ntrain));
}
