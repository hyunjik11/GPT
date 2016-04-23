library(abind)
#nsamples=numchains*numsamples
psiU=abind(out$psitrainU,out$psitestU,along=2); #nsamples by N by r
psiV=abind(out$psitrainV,out$psitestV,along=2); #nsamples by N by r
nsamples=dim(psiU)[1]
w=out$w; #nsamples by r by r
surfaces=array(0,dim=c(N,r,r))

for (i in 1:N){
  for (j in 1:r){
    for (k in 1:r){
      surfaces[i,j,k]=sum(psiU[,i,j]*psiV[,i,k]*w[,j,k])/nsamples
    }
  }
  if (i%%1000==0){
    cat(i," iterations out of ",N," done \n")
  }
}
data = read.table("cadata.txt", col.names=  c( "MedHouseVal", "MedInc", "HouseAge", "AveRooms", "AveBedrms",
                                                          "Population", "AveOccup", "long", "lat"))
data$MedHouseVal = log(data$MedHouseVal)
f=data.frame(data$long,data$lat,data$MedHouseVal)
long=f[perm,1]; lat=f[perm,2];

price.percentiles = quantile(surfaces,0:100/100)

par(mfrow=c(r,r))
for (j in 1:r){
  for (k in 1:r){
    cut.prices = cut(surfaces[,j,k],price.percentiles,include.lowest=TRUE)
    plot(lat,long,col=grey(109:10/130)[cut.prices],pch=20,cex=0.5,xlab="Latitude",ylab="Longitude")
  }
}
