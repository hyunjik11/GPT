addpath(genpath('/homes/hkim/Documents/GPstuff-4.6'));
num_workers=20;
POOL=parpool('local',num_workers);
% Load the data
x=h5read('/homes/hkim/GPT/PPdata.h5','/Xtrain');
y=h5read('/homes/hkim/GPT/PPdata.h5','/ytrain');
[n, D] = size(x);

% Now we will use the variational sparse approximation.

% First we create the GP structure. Notice here that if we do
% not explicitly set the priors for the covariance function
% parameters they are given a uniform prior.
lik = lik_gaussian('sigma2', 0.2^2);
gpcf = gpcf_sexp('lengthScale', ones(1,D), 'magnSigma2', 0.2^2);
gp=gp_set('lik',lik,'cf',gpcf); %exact gp

% Next we initialize the inducing inputs and set them in GP
% structure. We have to give a prior for the inducing inputs also,
% if we want to optimize them
m=100; %number of inducing pts.
data_samples=datasample(x,m*num_workers,1,'Replace',false);
parfor i=1:num_workers
X_u=data_samples((m*(i-1)+1):m*i,:); %each row specifies coordinates of an inducing point. here we randomly sample m data points
gp_var = gp_set('type', 'VAR', 'lik', lik, 'cf', gpcf,'X_u', X_u, 'jitterSigma2', 1e-4); %var_gp


% -----------------------------
% --- Conduct the inference ---

% Then we can conduct the inference. We can now optimize i) only
% the parameters, ii) both the parameters and the inducing inputs,
% or iii) only the inducing inputs. Which option is used is defined
% by a string that is given to the gp_pak, gp_unpak, gp_e and gp_g
% functions. The strings for the different options are:
% 'covariance+likelihood' (i), 'covariance+likelihood+inducing' (ii),
% 'inducing' (iii).
%

% Now you can choose, if you want to optimize only parameters or
% optimize simultaneously parameters and inducing inputs. Note that
% the inducing inputs are not transformed through logarithm when
% packed

% optimize parameters and inducing inputs
gp_var = gp_set(gp_var, 'infer_params', 'covariance+likelihood+inducing');
% optimize only parameters
%gp_var = gp_set(gp_var, 'infer_params', 'covariance+likelihood');           

opt=optimset('TolFun',1e-3,'TolX',1e-4,'MaxIter',1000);
% Optimize with the quasi-Newton method
% gp=gp_optim(gp,x,y,'opt',opt);
gp_var=gp_optim(gp_var,x,y,'opt',opt,'optimf',@fminscg); %can also use @fminlbfgs,@fminunc
% Set the options for the optimization
% fprintf('length_scale=[');
% fprintf('%s',num2str(gp_var.cf{1}.lengthScale));
% fprintf('] sf=%2.4f, signal_var=%2.4f \n',gp.cf{1}.magnSigma2,gp.lik.sigma2);
fprintf('length_scale=[');
fprintf('%s',num2str(gp_var.cf{1}.lengthScale));
fprintf('] sf=%2.4f, signal_var=%2.4f \n',gp_var.cf{1}.magnSigma2,gp_var.lik.sigma2);
end
delete(POOL);
% To optimize the parameters and inducing inputs sequentially uncomment the below lines
% $$$ iter = 1
% $$$ e = gp_e(w,gp_var,x,y)
% $$$ e_old = inf;
% $$$ while iter < 100 & abs(e_old-e) > 1e-3
% $$$     e_old = e;
% $$$     
% $$$     gp_var = gp_set(gp_var, 'infer_params', 'covariance+likelihood');  % optimize parameters and inducing inputs
% $$$     gp_var=gp_optim(gp_var,x,y,'opt',opt);
% $$$     gp_var = gp_set(gp_var, 'infer_params', 'inducing');  % optimize parameters and inducing inputs
% $$$     gp_var=gp_optim(gp_var,x,y,'opt',opt);
% $$$     e = gp_e(w,gp_var,x,y);
% $$$     iter = iter +1;
% $$$     [iter e]
% $$$ end
