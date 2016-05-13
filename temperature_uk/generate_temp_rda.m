%addpath(genpath('/Users/hyunjik11/Documents/gpml-matlab-v3.6-2015-07-07'))
addpath(genpath('/homes/hkim/Documents/gpml-matlab-v3.6-2015-07-07'))
load('/homes/hkim/TensorGP/src/uk_temp/temp1.mat');
hypspace=[hyp.cov(1),hyp.cov(2)];
hyptime=[hyp.cov(3),hyp.cov(4)];
x=[xtrain;xtest];
space=unique(x(:,1:2),'rows');
time=unique(x(:,3));
space_shuffled=space(ii_space,:);
time_shuffled=time(ii_temporal);
Kspace=covSEiso(hypspace,space);
Ktime=covSEiso(hyptime,time);
nspace=length(Kspace); ntime=length(Ktime);
Kspace=Kspace+1e-12*eye(nspace);
Ktime=Ktime+1e-12*eye(ntime);
[V,D]=eig(Kspace);phiU=V*sqrt(D);
[V,D]=eig(Ktime);phiV=V*sqrt(D);
sigma=exp(hyp.lik);
[~,spacetrain_idx]=ismember(xtrain(:,1:2),space,'rows');
[~,spacetest_idx]=ismember(xtest(:,1:2),space,'rows');
[~,timetrain_idx]=ismember(xtrain(:,3),time);
[~,timetest_idx]=ismember(xtest(:,3),time);
%phiU,phiV,space_shuffled,time_shuffled,xtrain,xtest,ytrain,ytest,spacetrain_idx,spacetest_idx,timetrain_idx,timetest_idx,ys,sqrt(signal_var)
%hdf5write('temp1.h5','/phiU',phiU,'/phiV',phiV,'/xtrain',xtrain,'/xtest',xtest,'/ytrain',ytrain,'/ytest',ytest,...
%    '/indtrainU',spacetrain_idx,'/indtestU',spacetest_idx,'/indtrainV',...
%    timetrain_idx,'/indtestV',timetest_idx,'/s',ys,'/sigma',sigma)
%%Convert to rda file as follows in R:
%library(h5)
%file=h5file("temp1.h5")
%phiU=file["phiU"][]
%phiV=file["phiV"]
%ytrain=file["ytrain"][]
%ytest=file["ytest"][]
%indtrainU=file["indtrainU"][]
%indtrainV=file["indtrainV"][]
%indtestU=file["indtestU"][]
%indtestV=file["indtestV"][]
%s=file["s"][]
%sigma=file["sigma"][]
%then add on the necessary constants (N,Ntrain,n1,n2 etc) before saving to rda
