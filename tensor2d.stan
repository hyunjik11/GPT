data {
  int<lower=1> N;
  int<lower=1> Ntrain;
  int<lower=1> n;
  int<lower=1> r;
  matrix[Ntrain,n] phitrainU;
  matrix[Ntrain,n] phitrainV;
  matrix[N-Ntrain,n] phitestU;
  matrix[N-Ntrain,n] phitestV;
  vector[Ntrain] ytrain;
  vector[N-Ntrain] ytest;
  real<lower=0> sigma;
  real<lower=0> ytrainStd;
}
parameters {
  matrix[n,r] U;
  matrix[n,r] V;
  matrix[r,r] w;
}
transformed parameters{
  matrix[Ntrain,r] psitrainU;
  matrix[Ntrain,r] psitrainV;
  vector[Ntrain] trainpred;
  psitrainU <- phitrainU*U;
  psitrainV <- phitrainV*V;
  for (i in 1:Ntrain)
	trainpred[i] <- psitrainU[i]*w*psitrainV[i]';
}
model {
  for (i in 1:n)
	  U[i] ~ normal(0,sqrt(1.0/r));
  for (i in 1:n)
	  V[i] ~ normal(0,sqrt(1.0/r));
  for (i in 1:r)  
	  w[i] ~ normal(0,1);
  ytrain ~ normal(trainpred,sigma);
}
generated quantities {
  vector[N-Ntrain] testpred;
  matrix[N-Ntrain,r] psitestU;
  matrix[N-Ntrain,r] psitestV;
  real trainRMSE;
  real testRMSE;
  psitestU <- phitestU*U;
  psitestV <- phitestV*V;
  for (i in 1:(N-Ntrain))
	testpred[i] <- psitestU[i]*w*psitestV[i]';
  trainRMSE <- ytrainStd*sqrt((ytrain-trainpred)'*(ytrain-trainpred)/Ntrain);
  testRMSE <- ytrainStd*sqrt((ytest-testpred)'*(ytest-testpred)/(N-Ntrain));
}
