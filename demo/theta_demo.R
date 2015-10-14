
N = 10;sigma=0.1;sigmaRBF = 5;set.seed(12)
seq_train = seq(0.5,by=1,length.out=N);seq_test = sort(N*runif(N^2/4))
X_train = cbind(rep(seq_train,N),rep(seq_train,each=N))
X_test = cbind(rep(seq_test,N/4),rep(seq_test,each=N/4))
f = GPdraw(rbind(X_train,X_test),sigmaRBF,1)
set.seed(12)
y = f + sigma*rnorm(length(f))
y_train = y[1:nrow(X_train)]
y_test = tail(y,nrow(X_test))
f_test = tail(f,nrow(X_test))
f_train = f[1:nrow(X_train)]
##################################################################################################################

library(xlsx)
# dyn.load("/homes/xlu/Dropbox/GP/GPT/src/U_phi.so")
data = read.xlsx("/homes/xlu/Dropbox/GP/GPT/CCPP/Folds5x2_pp.xlsx",sheetIndex=1,header=T)
# data = read.xlsx("C:/Users/Xiaoyu Lu/Dropbox/GP/GPT/CCPP/Folds5x2_pp.xlsx",sheetIndex=1,header=T)

input = as.matrix(data[,1:4]); output = data[,5]
input = apply(input,2,function(x) {(x-mean(x))/sd(x)})
y_mean = mean(data[1:5000,5]); y_std = sd(data[1:5000,5])
output = (output - y_mean)/y_std

X_train = input[1:5000,1:4]
X_test = input[5001:nrow(data),1:4]
y_train = output[1:5000];y_test = output[5001:length(output)]
# y_train_original = data[1:5000,5]
# y_test_original = data[5001:length(output),5]

sigma=0.2299;sigmaRBF=1.4332;generator=2;num_iterations=10;burnin=8
q=2^4




#############################PARAFAC####################################################
sigma=0.2299;sigmaRBF=1.4332;generator=2;n=5^4;

n_vec = c(5,50,80,100,200,300,450,500,625) ; RMSE_train_q=c(); RMSE_test_q = c(); timer_q = c()

for (j in 1:length(n_vec)) {
  
  n = n_vec[j]
  
  y_fit = function(W,X,n,generator) {
    D = ncol(X)
    Psi = sapply(1:nrow(X),function(i) {apply(feature(X[i,],n,sigmaRBF,generator),1,prod) })
    return(t(W) %*% Psi)
  }
    
  t0 = proc.time()
  
  res_parafac = parafac_inf(X_train,y_train,sigma,n,sigmaRBF,generator)
  y_fit_train = y_fit(res_parafac,X_train,n,generator)
  y_fit_test = y_fit(res_parafac,X_test,n,generator)
  
  RMSE_train_q[j] = y_std * sqrt(sum((y_fit_train-y_train)^2)/nrow(X_train))
  RMSE_test_q[j] = y_std * sqrt(sum((y_fit_test-y_test)^2)/nrow(X_test))
  timer_q[j] = (proc.time() - t0)[[1]]
  print(j)
}

par(mfrow=c(1,3))
plot(n_vec,RMSE_train_q,col=2,type="l",xlab="n",ylab="RMSE",ylim=c(3,8),
     main="RMSE on training set vs n")
abline(h=3.848); text(200,3.7,"RMSE exact = 3.848")
abline(h=4.035,col=3); text(400,4.2,col=3,"Full theta = 4.035")

plot(n_vec,RMSE_test_q,col=2,type="l",xlab="n",ylab="RMSE",ylim=c(3,8),
     main="RMSE on test set vs n")
abline(h=4.006); text(200,3.95,"RMSE exact = 4.006")
abline(h=4.157,col=3); text(400,4.30,col=3,"Full theta = 4.157")

plot(n_vec,timer_q,col=2,type="l",main="computational time vs n",xlab="n",ylab="time in seconds")
abline(h=17.101,col=3); text(400,18,col=3,"Full theta = 17.101")

#############################################################################################
#######################Full theta#################################################

n_vec = c(5)  ; RMSE_train=c(); RMSE_test = c(); timer = c()
for (j in 1:length(n_vec)) {
  
  n = n_vec[j]
  
  y_fit = function(theta,X,n,generator) {
    D = ncol(X)
    return(t(theta) %*% apply(X,1,function(x){Reduce(kronecker,alply(array(feature(x,n,sigmaRBF,generator),c(n,D)),2)) })
)
  }
  t0 = proc.time()

  res_theta = theta_inf(X_train,y_train,sigma,n,sigmaRBF,generator)
  y_fit_train = y_fit(res_theta,X_train,n,generator)
  y_fit_test = y_fit(res_theta,X_test,n,generator)
  
  RMSE_train[j] = y_std * sqrt(sum((y_fit_train-y_train)^2)/nrow(X_train))
  RMSE_test[j] = y_std * sqrt(sum((y_fit_test-y_test)^2)/nrow(X_test))
  timer[j] = (proc.time() - t0)[[1]]
  print(j)
}


######################################GP EXACT#####################################################
Kernel = function(x,y,sigmaRBF) {
  tmp = matrix(1,nrow(x),nrow(y))
  for (i in 1:nrow(x)) {
    for (j in 1:nrow(y)){
      tmp[i,j] = exp(-1/(2*sigmaRBF^2)*(t(x[i,]-y[j,])%*%(x[i,]-y[j,])))
    }
  } 
  return(tmp)
}
t0 = proc.time()
tmp = Kernel(X_train,X_train,sigmaRBF) + sigma^2 * diag(nrow(X_train))
mean_test = Kernel(X_test,X_train,sigmaRBF) %*% solve(tmp,y_train)
mean_train = Kernel(X_train,X_train,sigmaRBF) %*% solve(tmp,y_test)
RMSE_test_ex = round(sqrt(sum((mean_test-y_test)^2)/nrow(X_test)),4)
RMSE_train_ex = round(sqrt(sum((mean_train-y_train)^2)/nrow(X_train)),4)
timer_exact = (proc.time() - t0)[[1]]

tmp = solve(Kernel(X_train,X_train,sigmaRBF) + sigma^2 * diag(nrow(X_train))) %*% y_train
mean_test = Kernel(X_test,X_train,sigmaRBF) %*% tmp
mean_train = Kernel(X_train,X_train,sigmaRBF) %*% tmp

plot(y_test,type="l")
lines(mean_test,col=3)
RMSE_test_ex = round(sqrt(sum((mean_test-y_test)^2)/nrow(X_test)),4)
RMSE_train_ex = round(sqrt(sum((mean_train-y_train)^2)/nrow(X_train)),4)

