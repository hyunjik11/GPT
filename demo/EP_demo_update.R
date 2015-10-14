library(xlsx)
# dyn.load("src/U_phi.so")
data = read.xlsx("CCPP/Folds5x2_pp.xlsx",sheetIndex=1,header=T)

input = as.matrix(data[,1:4]); output = data[,5]
input = apply(input,2,function(x) {(x-mean(x))/sd(x)})
y_mean = mean(data[1:5000,5]); y_std = sd(data[1:5000,5])
output = (output - y_mean)/y_std

X_train = input[1:5000,1:4]
X_test = input[5001:nrow(data),1:4]
y_train = output[1:5000];y_test = output[5001:length(output)]


sigma=0.2299;n=10;r=10;sigmaRBF=1.4332;q=100;num_iterations=10;burnin=8
X=X_train;y=y_train;sigma=0.2299;n=20;q=100;generator=5;num_iterations=50;r=10;burnin=5

RMSE_train_r = c();RMSE_test_r = c();timer_r=c()
r_vec = c(5,10,20,30,49);n=50;q=100
for (i in 1:length(r_vec)) {
  r = r_vec[i]
  
  time0 = proc.time()
  res = GPT_inf(X_train,y_train,sigma,n,r,sigmaRBF,q,generator,num_iterations=15,burnin=10)
  
  y_fit = function(X,W_array,V_array,I,n,sigmaRBF,generator) {
    D = ncol(X);y_fit=matrix(,nrow(X),dim(W_array)[2])
    out_b = array(,c(n,nrow(X),ncol(X)))
    for (i in 1:nrow(X)){ out_b[,i,] = feature(X[i,],n,sigmaRBF,generator) }
    for (j in 1:dim(W_array)[2]) {
      
      tmp_array = array(,c(n,q,ncol(X)))
      for (d in 1:D){
        tmp_array[,,d] = V_array[,I[,d],d,j]
      }
      res = .C("M_func",as.double(tmp_array),as.double(out_b),as.integer(nrow(X)),as.integer(n),as.integer(q),as.integer(D),as.double(rep(0,n)),as.double(rep(0,n)),as.double(matrix(0,q,D)),
               as.double(rep(0,q)),as.double(matrix(0,nrow(X),q)))[[11]]
      UD = matrix(res,q,nrow(X))
      # UD = sapply(1:nrow(X),function(i) {apply(sapply((1:D),function(z) {  t(V_array[,I[,z],z,j]) %*% out_b[,i,z] }),1,prod) })      
      y_fit[,j] = t(W_array[,j]) %*% UD
    }
    return(rowMeans(y_fit))
  }
  
  
  y_fit_train = y_fit(X_train,res$W_array,res$V_array,res$I,n,sigmaRBF,generator)
  y_fit_test =  y_fit(X_test,res$W_array,res$V_array,res$I,n,sigmaRBF,generator)
  
  RMSE_train_r[i] = y_std* sqrt(sum((y_fit_train-y_train)^2)/nrow(X_train))
  RMSE_test_r[i] = y_std*sqrt(sum((y_fit_test-y_test)^2)/nrow(X_test))
  timer_r[i] = (proc.time() - time0)[[1]]
  print(paste("RMSE_train_r= ", RMSE_train_r))
}

par(mfrow=c(1,3))
plot(r_vec,RMSE_train_r,type="l",main="RMSE on training set vs r, n=100, q=300",xlab="r",
     ylim=range(c(RMSE_train_r,RMSE_test_r,3.848)));
abline(h=3.848,col=2); text(15,3.86,"RMSE exact = 3.848")
plot(r_vec,RMSE_test_r,type="l",main="RMSE on test set vs r, n=100, q=300",xlab="r",
     ylim=range(c(RMSE_train_r,RMSE_test_r,4.0063)))
abline(h=4.006,col=2); text(15,4.01,"RMSE exact = 4.006")
plot(r_vec,timer_r,type="l",main="computational time vs r, n=100, q=300",xlab="r",
     ylim=range(c(timer_r,1215.832)))
abline(h=1215.832,col=2); text(15,1230,"timer exact = 1215.832")

save(RMSE_train_r,file="RMSE_train_r_file")
save(RMSE_test_r,file="RMSE_test_r_file")
save(timer_r,file="timer_r_file")

############################################################################################################
#############################################################################################################

#investigate relationship with n
X=X_train;y=y_train;sigma=0.2299;sigmaRBF=1.4332;q=100;generator=5;r=10;

RMSE_train_n = c();RMSE_test_n = c();timer_n=c()
n_vec = c(30,50,80,100,150)
for (i in 1:length(n_vec)) {
  n = n_vec[i]
  
  time0 = proc.time()
  res = GPT_inf(X_train,y_train,sigma,n,r,sigmaRBF,q,generator,num_iterations=15,burnin=10)
  
  y_fit = function(X,W_array,V_array,I,n,sigmaRBF,generator) {
    D = ncol(X);y_fit=matrix(,nrow(X),dim(W_array)[2])
    out_b = array(,c(n,nrow(X),ncol(X)))
    for (i in 1:nrow(X)){ out_b[,i,] = feature(X[i,],n,sigmaRBF,generator) }
    for (j in 1:dim(W_array)[2]) {
      
      tmp_array = array(,c(n,q,ncol(X_train)))
      for (d in 1:D){
        tmp_array[,,d] = V_array[,I[,d],d,j]
      }
      res = .C("M_func",as.double(tmp_array),as.double(out_b),as.integer(nrow(X)),as.integer(n),as.integer(q),as.integer(D),as.double(rep(0,n)),as.double(rep(0,n)),as.double(matrix(0,q,D)),
               as.double(rep(0,q)),as.double(matrix(0,nrow(X),q)))[[11]]
      UD = matrix(res,q,nrow(X))
      # UD = sapply(1:nrow(X),function(i) {apply(sapply((1:D),function(z) {  t(V_array[,I[,z],z,j]) %*% out_b[,i,z] }),1,prod) })      
      y_fit[,j] = t(W_array[,j]) %*% UD
    }
    return(rowMeans(y_fit))
  }
  
  
  y_fit_train = y_fit(X_train,res$W_array,res$V_array,res$I,n,sigmaRBF,generator)
  y_fit_test =  y_fit(X_test,res$W_array,res$V_array,res$I,n,sigmaRBF,generator)
  
  RMSE_train_n[i] = y_std* sqrt(sum((y_fit_train-y_train)^2)/nrow(X_train))
  RMSE_test_n[i] = y_std*sqrt(sum((y_fit_test-y_test)^2)/nrow(X_test))
  
  timer_n[i] = (proc.time() - time0)[[1]]
}



par(mfrow=c(1,3))
plot(n_vec,RMSE_train_n,type="l",main="RMSE on training set vs n, r=30, q=300",xlab="n",
     ylim=range(c(RMSE_train_n,RMSE_test_n,3.848)));
abline(h=3.848,col=2); text(100,3.86,"RMSE exact = 3.848")
plot(n_vec,RMSE_test_n,type="l",main="RMSE on test set vs n, r=30, q=300",xlab="n",
     ylim=range(c(RMSE_train_n,RMSE_test_n,4.0063)))
abline(h=4.006,col=2); text(100,4.01,"RMSE exact = 4.006")
plot(n_vec,timer_n,type="l",main="computational time vs n, r=30, q=300",xlab="n",
     ylim=range(c(timer_n,1215.832)))
abline(h=1215.832,col=2); text(100,1230,"timer exact = 1215.832")



#####################################################################################################
#investigate relationship with q
X=X_train;y=y_train;sigma=0.2299;sigmaRBF=1.4332;q=100;generator=5;r=30;n=50

RMSE_train_q = c();RMSE_test_q = c();timer_q=c()
q_vec = c(10,100,200,300)
for (i in 1:4) {
  q = q_vec[i]
  
  time0 = proc.time()
  res = GPT_inf(X_train,y_train,sigma,n,r,sigmaRBF,q,generator,num_iterations=15,burnin=10)
  
  y_fit = function(X,W_array,V_array,I,n,sigmaRBF,generator) {
    D = ncol(X);y_fit=matrix(,nrow(X),dim(W_array)[2])
    out_b = array(,c(n,nrow(X),ncol(X)))
    for (i in 1:nrow(X)){ out_b[,i,] = feature(X[i,],n,sigmaRBF,generator) }
    for (j in 1:dim(W_array)[2]) {
      
      tmp_array = array(,c(n,q,ncol(X_train)))
      for (d in 1:D){
        tmp_array[,,d] = V_array[,I[,d],d,j]
      }
      res = .C("M_func",as.double(tmp_array),as.double(out_b),as.integer(nrow(X)),as.integer(n),as.integer(q),as.integer(D),as.double(rep(0,n)),as.double(rep(0,n)),as.double(matrix(0,q,D)),
               as.double(rep(0,q)),as.double(matrix(0,nrow(X),q)))[[11]]
      UD = matrix(res,q,nrow(X))
      # UD = sapply(1:nrow(X),function(i) {apply(sapply((1:D),function(z) {  t(V_array[,I[,z],z,j]) %*% out_b[,i,z] }),1,prod) })      
      y_fit[,j] = t(W_array[,j]) %*% UD
    }
    return(rowMeans(y_fit))
  }
  
  
  y_fit_train = y_fit(X_train,res$W_array,res$V_array,res$I,n,sigmaRBF,generator)
  y_fit_test =  y_fit(X_test,res$W_array,res$V_array,res$I,n,sigmaRBF,generator)
  
  RMSE_train_q[i] = y_std* sqrt(sum((y_fit_train-y_train)^2)/nrow(X_train))
  RMSE_test_q[i] = y_std*sqrt(sum((y_fit_test-y_test)^2)/nrow(X_test))
  
  timer_q[i] = (proc.time() - time0)[[1]]
}


par(mfrow=c(1,3))
plot(q_vec,RMSE_train_q,type="l",main="RMSE on training set vs q, r=30, n=100",xlab="q",
     ylim=range(c(RMSE_train_q,RMSE_test_q,3.848)),ylab="RMSE");
abline(h=3.848,col=2); text(200,3.86,"RMSE exact = 3.848")
plot(q_vec,RMSE_test_q,type="l",main="RMSE on test set vs q, r=30, n=100",xlab="q",
     ylim=range(c(RMSE_train_q,RMSE_test_q,4.0063)),ylab="RMSE");
abline(h=4.006,col=2); text(200,4.01,"RMSE exact = 4.006")
plot(q_vec,timer_q,type="l",main="computational time vs q, r=30, n=100",xlab="q",
     ylim=range(c(timer_q,1215.832)),ylab="time in seconds");
abline(h=1215.832,col=2); text(200,1230,"timer exact = 1215.832")

