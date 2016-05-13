library(R.matlab)
data=readMat("/homes/hkim/TensorGP/src/uk_temp/temp1.mat")
ytrain=mydata$ytrain
ytest=mydata$ytest
s=mydata$ys
lspace=exp(mydata$hyp[1][[1]][1,1])
sfspace=exp(mydata$hyp[1][[1]][2,1])
ltime=exp(mydata$hyp[1][[1]][3,1])
sftime=exp(mydata$hyp[1][[1]][4,1])
sigma=exp(mydata$hyp[2][[1]][1,1])
#say space,temporal are vectors where each row is the space/time coordinates, in the right order

n=25 #number of RFF
#phiU for space
D=2
Z=rnorm(n*D)/lspace
Z=matrix(Z,nrow=D)
b=2*pi*runif(n)
phiU=matrix(0,length(space),n)
for (i in 1:length(space)){
  phiU[i,] = cos(as.matrix(space[i,]) %*% Z+t(b))
}
phiU=sfspace*sqrt(2/n)*phiU

#phiV for time
D=1
Z=rnorm(n*D)/ltime
b=2*pi*runif(n)
phiV=matrix(0,length(temporal),n)
for (i in 1:length(temporal)){
  phiV[i,] = cos(temporal[i]*Z+t(b))
}
phiV=sftime*sqrt(2/n)*phiV


