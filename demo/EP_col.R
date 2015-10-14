
RMSE_train_r_col = c();RMSE_test_r_col = c();timer_r_col=c()
r_vec = c(5,10,20,30,50);n=50;q=100
for (i in 1:length(r_vec)) {
  r = r_vec[i]
  
  time0 = proc.time()
  res = GPT_infcol(X_train,y_train,sigma,n,r,sigmaRBF,q,generator,num_iterations=30,burnin=20)
  
  y_fit = function(X,W_array,V_array,I,n,sigmaRBF,generator) {
    D = ncol(X);y_fit=matrix(,nrow(X),dim(W_array)[2])
    out_b = array(,c(n,nrow(X),ncol(X)))
    for (i in 1:nrow(X)){ out_b[,i,] = feature(X[i,],n,sigmaRBF,generator) }
    for (j in 1:dim(W_array)[2]) {
      UD = sapply(1:nrow(X),function(i) {apply(sapply((1:D),function(z) {  t(V_array[,I[,z],z,j]) %*% out_b[,i,z] }),1,prod) })      
      y_fit[,j] = t(W_array[,j]) %*% UD
    }
    return(rowMeans(y_fit))
  }
  
  y_fit_train = y_fit(X_train,res$W_array,res$V_array,res$I,n,sigmaRBF,generator)
  y_fit_test =  y_fit(X_test,res$W_array,res$V_array,res$I,n,sigmaRBF,generator)
  
  RMSE_train_r_col[i] = y_std* sqrt(sum((y_fit_train-y_train)^2)/nrow(X_train))
  RMSE_test_r_col[i] = y_std*sqrt(sum((y_fit_test-y_test)^2)/nrow(X_test))
  timer_r_col[i] = (proc.time() - time0)[[1]]
}


# save(RMSE_train_r_col,file="RMSE_train_r_EPcol")
# save(RMSE_test_r_col,file="RMSE_test_r_EPcol")
# save(timer_r_col,file="timer_r_EPcol")
# 
# 
# load("RMSE_train_r_file")   #### learn matrices
# load("RMSE_test_r_file")
# load("timer_r_file")
# 

par(mfrow=c(1,3))
plot(r_vec,RMSE_train_r_col,type="l",col=3,main="RMSE on training set vs r, n=50, q=100",xlab="r",
     ylim=range(c(RMSE_train_r,RMSE_test_r,RMSE_train_r_col,3.848)),ylab="RMSE");
lines(r_vec,RMSE_train_r,col=2)
abline(h=3.848); text(15,3.855,"GP exact = 3.848")
legend("topright",c("sparse tucker","sparse tucker independent column"),col=c(2,3),lwd=c(1,1))

plot(r_vec,RMSE_test_r_col,type="l",col=3,main="RMSE on test set vs r, n=50, q=100",xlab="r",
     ylim=range(c(RMSE_train_r,RMSE_test_r,RMSE_test_r_col,4.0063)),ylab="RMSE");
lines(r_vec,RMSE_test_r,col=2)
abline(h=4.006); text(15,4.01,"GP exact = 4.006")
legend("topright",c("sparse tucker","sparse tucker independent column"),col=c(2,3),lwd=c(1,1))

plot(r_vec,timer_r_col,type="l",col=3,main="computational time vs r, n=50, q=100",xlab="r",
     ylim=range(c(timer_r,timer_r_col,1215.832)),ylab="RMSE");
lines(r_vec,timer_r,col=2)
abline(h=1215.832); text(15,1240,"GP exact = 1215.832")
legend("topright",c("sparse tucker","sparse tucker independent column"),col=c(2,3),lwd=c(1,1))

############################################################################################################
#############################################################################################################

#investigate relationship with n
X=X_train;y=y_train;sigma=0.2299;sigmaRBF=1.4332;q=100;generator=123;r=20;

RMSE_train_n_col = c();RMSE_test_n_col = c();timer_n_col=c()
n_vec = c(30,50,80,100,150,200)
for (i in 1:length(n_vec)) {
  n = n_vec[i]
  
  time0 = proc.time()
  res = GPT_infcol(X_train,y_train,sigma,n,r,sigmaRBF,q,generator,num_iterations=15,burnin=10)
  
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
  
  RMSE_train_n_col[i] = y_std* sqrt(sum((y_fit_train-y_train)^2)/nrow(X_train))
  RMSE_test_n_col[i] = y_std*sqrt(sum((y_fit_test-y_test)^2)/nrow(X_test))
  
  timer_n_col[i] = (proc.time() - time0)[[1]]
}

save(RMSE_train_n_col,file="RMSE_train_n_EPcol")
save(RMSE_test_n_col,file="RMSE_test_n_EPcol")
save(timer_n_col,file="timer_n_EPcol")
# load("RMSE_train_n_file")
# load("RMSE_test_n_file")
# load("timer_n_file")


par(mfrow=c(1,3))
plot(n_vec,RMSE_train_n_col,type="l",col=3,main="RMSE on training set vs n, r=20, q=100",xlab="n",
     ylim=range(c(RMSE_train_n,RMSE_test_n,RMSE_train_n_col,3.848)),ylab="RMSE");
lines(c(30,50,80,100,150,200),RMSE_train_n,col=2)
abline(h=3.848); text(100,3.855,"GP exact = 3.848")
legend("topright",c("sparse tucker","sparse tucker independent column"),col=c(2,3),lwd=c(1,1))

plot(n_vec,RMSE_test_n_col,type="l",col=3,main="RMSE on test set vs n, n=50, q=100",xlab="n",
     ylim=range(c(RMSE_train_r,RMSE_test_r,RMSE_test_n_col,4.0063)),ylab="RMSE");
lines(c(30,50,80,100,150,200),RMSE_test_r,col=2)
abline(h=4.006); text(15,4.01,"GP exact = 4.006")
legend("topright",c("sparse tucker","sparse tucker independent column"),col=c(2,3),lwd=c(1,1))

plot(n_vec,timer_n_col,type="l",col=3,main="computational time vs n, n=50, q=100",xlab="n",
     ylim=range(c(timer_r,timer_n_col,1215.832)),ylab="RMSE");
lines(c(30,50,80,100,150,200),timer_r,col=2)
abline(h=1215.832); text(15,1240,"GP exact = 1215.832")
legend("topright",c("sparse tucker","sparse tucker independent column"),col=c(2,3),lwd=c(1,1))

############################################################################################################
#############################################################################################################

#investigate relationship with q
X=X_train;y=y_train;sigma=0.2299;sigmaRBF=1.4332;q=100;generator=5;r=30;n=100

RMSE_train_q_col = c();RMSE_test_q_col = c();timer_q_col=c()
q_vec = c(80,100,200,300)
for (i in 1:length(q_vec)) {
  q = q_vec[i]
  
  time0 = proc.time()
  res = GPT_infcol(X_train,y_train,sigma,n,r,sigmaRBF,q,generator,num_iterations=15,burnin=10)
  
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
  
  RMSE_train_q_col[i] = y_std* sqrt(sum((y_fit_train-y_train)^2)/nrow(X_train))
  RMSE_test_q_col[i] = y_std*sqrt(sum((y_fit_test-y_test)^2)/nrow(X_test))
  
  timer_q_col[i] = (proc.time() - time0)[[1]]
}

save(RMSE_train_q_col,file="RMSE_train_q_EPcol")
save(RMSE_test_q_col,file="RMSE_test_q_EPcol")
save(timer_q_col,file="timer_q_EPcol")


par(mfrow=c(1,3))
plot(q_vec,RMSE_train_q_col,type="l",main="RMSE on training set vs q, r=30, n=100",xlab="q",
     ylim=range(c(RMSE_train_q_col,RMSE_test_q_col,3.848)),ylab="RMSE");
abline(h=3.848,col=2); text(200,3.86,"RMSE exact = 3.848")
plot(q_vec,RMSE_test_q_col,type="l",main="RMSE on test set vs q, r=30, n=100",xlab="q",
     ylim=range(c(RMSE_train_q_col,RMSE_test_q_col,4.0063)),ylab="RMSE");
abline(h=4.006,col=2); text(200,4.01,"RMSE exact = 4.006")
plot(q_vec,timer_q_col,type="l",main="computational time vs q, r=30, n=100",xlab="q",
     ylim=range(c(timer_q_col,1215.832)),ylab="time in seconds");
abline(h=1215.832,col=2); text(200,1230,"timer exact = 1215.832")

