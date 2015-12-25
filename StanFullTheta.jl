using Stan, Mamba, Compat

old=pwd()
ProjDir="/Users/hyunjik11/Documents/Stan/Examples";
cd(ProjDir)
fullthetastanmodel="
data {
  int<lower=1> N;
  int<lower=1> n;
  matrix[N,n] phi;
  vector[N] y;
  real<lower=0> sigma;
  real<lower=0.0001> sigma_theta;
}
parameters {
  vector[n] theta;
}
model {
  theta ~ normal(0,sigma_theta);
  y ~ normal(phi*theta,sigma);
}
generated quantities {
  vector[N] pred;
  real RMSE;
  pred <- phi * theta;
  RMSE <- sqrt((y-pred)'*(y-pred)/N);
}
"
fullthetadata= [
                      @Compat.Dict("N" => Ntrain,
                                   "n" => n,
                                   "phi" => phitrain',
                                   "y" => ytrain,
                                   "sigma" => sigma,
                                   "sigma_theta" => sigma_theta)
                      ];
stanmodel=Stanmodel(adapt=1,update=1,name="fulltheta",model=fullthetastanmodel);
sim1=stan(stanmodel,fullthetadata,ProjDir,CmdStanDir="/Users/hyunjik11/Documents/Stan/cmdstan-2.9.0")
nodesubset=["lp__","accept_stat__","theta.1", "theta.2", "theta.3", "theta.4", "theta.5"]
sim=sim1[:, nodesubset, :]

describe(sim)
try
  gelmandiag(sim1, mpsrf=true, transform=true) |> display
catch e
  #println(e)
  gelmandiag(sim, mpsrf=false, transform=true) |> display
end

@osx ? if isdefined(Main, :JULIA_SVG_BROWSER) && length(JULIA_SVG_BROWSER) > 0
        for i in 1:4
          isfile("$(stanmodel.name)-summaryplot-$(i).svg") &&
            run(`open -a $(JULIA_SVG_BROWSER) "$(stanmodel.name)-summaryplot-$(i).svg"`)
        end
      end : println()

