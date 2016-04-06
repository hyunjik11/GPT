addpath(genpath('/homes/hkim/Documents/GPstuff-4.6'));
%addpath(genpath('/Users/hyunjik11/Documents/GPstuff'));
%num_workers=10;
%POOL=parpool('local',num_workers);
% % Load the data
x=h5read('/homes/hkim/GPT/PPdata.h5','/Xtrain');
y=h5read('/homes/hkim/GPT/PPdata.h5','/ytrain');
%x=h5read('PPdata_full.h5','/Xtrain');
%y=h5read('PPdata_full.h5','/ytrain');
x=x(1:500,:); y=y(1:500); %only use 500 pts for faster computation.
%% PPfull hyperparams
length_scale=[1.3978 0.0028 2.8966 7.5565];
sigma_RBF2=0.8333; 
signal_var=0.0195;
[n, D] = size(x);
lik = lik_gaussian('sigma2', 0.2^2);
gpcf = gpcf_sexp('lengthScale', ones(1,D), 'magnSigma2', 0.2^2);
m=10;
X_u=datasample(x,m,1,'Replace',false); %each row specifies coordinates of an inducing point. here we randomly sample m data points
gp_var = gp_set('type', 'VAR', 'lik', lik, 'cf', gpcf,'X_u', X_u); %var_gp
gp_var = gp_set(gp_var, 'infer_params', 'covariance+likelihood+inducing');
% optimize only parameters
%gp_var = gp_set(gp_var, 'infer_params', 'covariance+likelihood');           

opt=optimset('TolFun',1e-3,'TolX',1e-4,'Display','iter','MaxIter',1000);
% Optimize with the quasi-Newton method
%gp=gp_optim(gp,x,y,'opt',opt);
tic()
gp_var=gp_optim(gp_var,x,y,'opt',opt,'optimf',@fminscg); %can also use @fminlbfgs,@fminunc
toc()
fprintf('m=%d,-l=%2.4f \n',10*m,gp_e([],gp_var,x,y));
%end
[temp,nll]=gp_e([],gp_var,x,y);
fprintf('-l=%2.4f;',nll);
fprintf('length_scale=[');
fprintf('%s',num2str(gp_var.cf{1}.lengthScale));
fprintf('];sigma_RBF2=%2.4f;signal_var=%2.4f \n',gp_var.cf{1}.magnSigma2,gp_var.lik.sigma2);