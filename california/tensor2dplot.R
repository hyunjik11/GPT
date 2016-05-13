#to save plot:
#1. png('image_name.png',width=1800,height=1200)
#2. source this script
#3. dev.off()

library(abind)
#nsamples=numchains*numsamples
psiU=abind(out$psitrainU,out$psitestU,along=2); #nsamples by N by r
psiV=abind(out$psitrainV,out$psitestV,along=2); #nsamples by N by r
nsamples=dim(psiU)[1]
w=out$w; #nsamples by r by r
psiU=psiU[nsamples,,];
psiV=psiV[nsamples,,];
w=w[nsamples,,];
#just use last sample
surfaces=array(0,dim=c(N,r,r))

for (i in 1:N){
  for (j in 1:r){
    for (k in 1:r){
      surfaces[i,j,k]=psiU[i,j]*psiV[i,k]*w[j,k]
    }
  }
  #if (i%%1000==0){
  #  cat(i," iterations out of ",N," done \n")
  #}
}
data = read.table("cadata.txt", col.names=  c( "MedHouseVal", "MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "lat", "long"))
data$MedHouseVal = log(data$MedHouseVal)
f=data.frame(data$lat,data$long,data$MedHouseVal)
lat=f[perm,1]; long=f[perm,2]; loghouseval=f[perm,3];

#colour.grid = quantile(surfaces,0:100/100) #different color for each percentile
#colour.grid=seq(min(surfaces),max(surfaces),length.out=101) #uniform colouring

pred=apply(surfaces,1,sum)*sd(loghouseval)+mean(loghouseval) #predictions for loghouseprices

colour.grid =seq(8,16,length.out=101) #uniform colouring 

# zoom in on bay area
#indices=intersect(which(lat>36 & lat<39),which(long>-123 & long< -121))
# zoom in on LA area
#indices=intersect(which(lat>32 & lat<35),which(long>-120 & long< -116))

#restrict to zoomed in area
#long=long[indices]
#lat=lat[indices]
#surfaces=surfaces[indices,,]

#plot true house prices
# png('cali_true_loghouseprices.png',width=1800,height=1200)
# cut.prices=cut(loghouseval,colour.grid,include.lowest=TRUE)
# plot(long,lat,col=colorRampPalette(c("blue","red"))(100)[cut.prices],pch=20,cex=0.5)
# title(xlab="Longitude",ylab="Latitude")
# dev.off()

#plot predictions
png('cali_prediction_tensor2d_5r.png',width=1800,height=1200)
cut.prices=cut(pred,colour.grid,include.lowest=TRUE)
plot(long,lat,col=colorRampPalette(c("blue","red"))(100)[cut.prices],pch=20,cex=0.5)
title(xlab="Longitude",ylab="Latitude")
dev.off()

# png('cali_tensor2d_2r_unif.png',width=1800,height=1200)
# par(mfrow=c(r,r),oma = c(5,4,0,0) + 0.1,mar = c(0,0,1,1) + 0.1)
# for (j in 1:r){
#   for (k in 1:r){
#     cut.prices = cut(surfaces[,j,k],colour.grid,include.lowest=TRUE)
# 	if (k%%r==1 && j==r){
#     	plot(long,lat,col=colorRampPalette(c("blue","red"))(100)[cut.prices],pch=20,cex=0.5)
# 	}
# 	else if (k%%r==1){
# 		plot(long,lat,col=colorRampPalette(c("blue","red"))(100)[cut.prices],pch=20,cex=0.5,xaxt='n')
# 	}	
# 	else if (j==r){
# 		plot(long,lat,col=colorRampPalette(c("blue","red"))(100)[cut.prices],pch=20,cex=0.5,yaxt='n')
# 	}
# 	else {
# 		plot(long,lat,col=colorRampPalette(c("blue","red"))(100)[cut.prices],pch=20,cex=0.5,xaxt='n',yaxt='n')
# 	}
#   }
# }
# title(xlab="Longitude",ylab="Latitude",outer=TRUE)
# dev.off()
