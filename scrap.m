addpath(genpath('/homes/hkim/Documents/GPstuff-4.6'));
%addpath(genpath('/Users/hyunjik11/Documents/GPstuff'));
num_workers=10;
%POOL=parpool('local',num_workers);
%%Load the data
load mauna.txt
z=mauna(:,2) ~= -99.99;
x=mauna(z,1); y=mauna(z,2);

%% initial hyperparameters
se1_l=4;
se1_s2=6;
lin_l=5;
lin_loc=
sigma_RBF2=0.8333; 
signal_var=0.0195;
[n, D] = size(x);


% Now we will use the variational sparse approximation.

% First we create the GP structure. Notice here that if we do
% not explicitly set the priors for the covariance function
% parameters they are given a uniform prior.
% lik = lik_gaussian('sigma2', 0.2^2);
% gpcf = gpcf_sexp('lengthScale', ones(1,D), 'magnSigma2', 0.2^2);
lik = lik_gaussian('sigma2', signal_var);
gpcf = gpcf_sexp('lengthScale', length_scale, 'magnSigma2', sigma_RBF2);
gp=gp_set('lik',lik,'cf',gpcf); %exact gp
[K,C]=gp_trcov(gp,x);

%opt=optimset('TolFun',1e-3,'TolX',1e-4,'Display','off','MaxIter',1000);%,'Display','off');
%gp=gp_optim(gp,x,y,'opt',opt);
%[temp,nll]=gp_e([],gp_var,x,y);
%nll_values(i)=nll; length_scale1_values(i)=gp_var.cf{1}.lengthScale(1);
%sigmaRBF2_values(i)=gp_var.cf{1}.magnSigma2; signal_var_values(i)=gp_var.lik.sigma2;
%fprintf('-l=%2.4f;',nll);
%fprintf('length_scale=[');
%fprintf('%s',num2str(gp.cf{1}.lengthScale));
%fprintf('];sigma_RBF2=%2.4f;signal_var=%2.4f \n',gp.cf{1}.magnSigma2,gp.lik.sigma2);
%fprintf('-l=%2.4f;',gp_e([],gp_var,x,y));
%fprintf('length_scale=[');
%fprintf('%s',num2str(gp_var.cf{1}.lengthScale));
%fprintf('];sigma_RBF2=%2.4f;signal_var=%2.4f \n',gp_var.cf{1}.magnSigma2,gp_var.lik.sigma2);
