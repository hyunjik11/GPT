@everywhere using Stan, Mamba, Compat

@everywhere old=pwd()
@everywhere ProjDir="/Users/hyunjik11/Documents/Stan/Examples/FullTheta";
@everywhere cd(ProjDir)
fullthetastanmodel="
data {
  int<lower=1> N;
  int<lower=1> Ntrain;
  int<lower=1> n;
  matrix[Ntrain,n] phitrain;
  matrix[N-Ntrain,n] phitest;
  vector[Ntrain] ytrain;
  vector[N-Ntrain] ytest;
  real<lower=0> sigma;
  real<lower=0> sigma_theta;
  real<lower=0> ytrainStd;
}
parameters {
  vector[n] theta;
}
model {
  theta ~ normal(0,sigma_theta);
  ytrain ~ normal(phitrain*theta,sigma);
}
generated quantities {
  vector[N] trainpred;
  vector[N] testpred;
  real trainRMSE;
  real testRMSE;
  trainpred <- phitrain * theta;
  testpred <- phitest * theta;
  trainRMSE <- ytrainStd*sqrt((ytrain-trainpred)'*(ytrain-trainpred)/Ntrain);
  testRMSE <- ytrainStd*sqrt((ytest-testpred)'*(ytest-testpred)/(N-Ntrain));
}
"
fullthetadata= [
                @Compat.Dict("N" => N,
                             "Ntrain" => Ntrain,  
                             "n" => n,
                             "phitrain" => phitrain',
                             "phitest" => phitest',
                             "ytrain" => ytrain,
                             "ytest" => ytest,
                             "sigma" => sigma,
                             "sigma_theta" => sigma_theta,
                             "ytrainStd" => ytrainStd)
                      ]
stanmodel=Stanmodel(thin=50,update=200,name="fulltheta",model=fullthetastanmodel);
tic();
sim1=stan(stanmodel,fullthetadata,ProjDir,CmdStanDir="/Users/hyunjik11/Documents/Stan/cmdstan-2.9.0")
toc();
nodesubset=["lp__","accept_stat__","theta.1", "theta.2", "theta.3", "theta.4", "theta.5"]
sim=sim1[:, nodesubset, :];

p=Mamba.plot(sim, [:trace, :mean, :density, :autocor], legend=true);
draw(p, ncol=4, filename="summaryplot", fmt=:pdf);

