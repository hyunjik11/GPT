data {
  int<lower=1> N;
  int<lower=1> Ntrain;
  int<lower=1> n1;
  int<lower=1> n2;
  int<lower=1> r;
  matrix[Ntrain,n1] phitrainU;
  matrix[Ntrain,n2] phitrainV;
  matrix[N-Ntrain,n1] phitestU;
  matrix[N-Ntrain,n2] phitestV;
  vector[Ntrain] ytrain;
  real<lower=0> sigma;
}
parameters {
  matrix[n1,r] U;
  matrix[n2,r] V;
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
  for (i in 1:n1)
	  U[i] ~ normal(0,sqrt(1.0/r));
  for (i in 1:n2)
	  V[i] ~ normal(0,sqrt(1.0/r));
  for (i in 1:r)  
	  w[i] ~ normal(0,1);
  ytrain ~ normal(trainpred,sigma);
}
generated quantities {
  vector[N-Ntrain] testpred;
  matrix[N-Ntrain,r] psitestU;
  matrix[N-Ntrain,r] psitestV;
  psitestU <- phitestU*U;
  psitestV <- phitestV*V;
  for (i in 1:(N-Ntrain))
	testpred[i] <- psitestU[i]*w*psitestV[i]';
}
