
data = read.csv("/homes/xlu/Dropbox/GP/GPT/CASP.csv",header=T)
N_train = 100 #round(nrow(data)/2); N_test = nrow(data) - N_train
input = as.matrix(data[,2:10]); output = data[,1]
input = apply(input,2,function(x) {(x-mean(x))/sd(x)})
y_mean = mean(data[1:N_train,1]); y_std = sd(data[1:N_train,1])
output = (output - y_mean)/y_std

X_train = input[1:N_train,]
X_test = input[(N_train+1):nrow(data),]
y_train = output[1:N_train];y_test = output[(N_train+1):length(output)]
################################################################################################
sigmaRBF = 1.5;generator = 100


Kernel = function(x,y,sigmaRBF) {
  tmp = matrix(1,nrow(x),nrow(y))
  for (i in 1:nrow(x)) {
    for (j in 1:nrow(y)){
      tmp[i,j] = exp(-1/(2*sigmaRBF^2)*(t(x[i,]-y[j,])%*%(x[i,]-y[j,])))
    }
  } 
  return(tmp)
}


K = Kernel(X_train,X_train,sigmaRBF)


n_vec = c(5,10,20,30,50,100,200);RMSE_K = c();RMSE_Kprod = c()
for (k in 1:length(n_vec)) {
  n = n_vec[k];D=ncol(X_train);N_train=nrow(X_train)
  K1 = matrix(0,N_train,N_train);K2 = matrix(0,N_train,N_train)
  for (i in 1:N_train) {
    for (j in (i:N_train)) {
      K1[i,j] = prod(sapply(1:D,function(z) {
        t(feature(X_train[i,],n,sigmaRBF,generator)[,z]) %*% feature(X_train[j,],n,sigmaRBF,generator)[,z]
      }))
      # K2[i,j] = t(fea_notensor(X_train[i,],n,sigmaRBF,generator)) %*% fea_notensor(X_train[j,],n,sigmaRBF,generator)
      }
    }
    
  
  K1 = K1 + t(K1) - diag(N_train) * diag(K1)
  K2 = K2 + t(K2) - diag(N_train) * diag(K2)
  
  RMSE_Kprod[k] = sqrt(sum((K-K1)^2)/length(K))
  RMSE_K[k] = sqrt(sum((K-K2)^2)/length(K))
  print(k)
}

plot(n_vec,RMSE_K,type="l",xlab="n",ylab="RMSE",main="RMSE of Kernel",ylim=range(RMSE_K,RMSE_Kprod))
lines(n_vec,RMSE_Kprod,col=2);legend("topright",c("no tensor","tensor"),col=c(1,2),lwd=c(1,1))

heatmap(K,xlab="x1",ylab="x2",main="heatmap")
heatmap(K1,xlab="x1",ylab="x2",main="Fitted heatmap")
heatmap(K2,xlab="x1",ylab="x2",main="Fitted heatmap tensor")

