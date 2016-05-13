long_mean=mean(f[1:Ntrain,2]); lat_mean=mean(f[1:Ntrain,1])
long_sd=sd(f[1:Ntrain,2]); lat_sd=sd(f[1:Ntrain,1])
npts=500
long_range=seq(min(f[,2]),max(f[,2]),length.out=npts); lat_range=seq(min(f[,1]),max(f[,1]),length.out=npts)
long_range_norm=(long_range-long_mean)/long_sd; lat_range_norm=(lat_range-lat_mean)/lat_sd
long_phi=matrix(0,npts,n); lat_phi=matrix(0,npts,n)
for(i in 1:npts) {
  lat_phi[i,]=cos(t(Z1*long_range_norm[i]+b1))
  long_phi[i,]=cos(t(Z2*lat_range_norm[i]+b2))
}
long_phi=sqrt(2/n)*long_phi;lat_phi=sqrt(2/n)*lat_phi

U=out$U; V=out$V
U=U[nsamples,,]; V=V[nsamples,,]
lat_psi=lat_phi%*%U; long_psi=long_phi%*%V

png('cali_double_tensor2d_5r.png',width=1800,height=1200)
cl=rainbow(r)
for (j in 1:r){
  if (j==1){plot(long_range,long_psi[,j],col=cl[j],type="l",xaxt='n',yaxt='n',xlab=NA,ylab=NA,
       xlim=c(min(f[,2])-3,max(f[,2])),ylim=c(min(long_psi),3*max(long_psi)-2*min(long_psi)))}
  else {
    lines(long_range,long_psi[,j],col=cl[j],xaxt='n',yaxt='n',xlab=NA,ylab=NA,
         xlim=c(min(f[,2])-3,max(f[,2])),ylim=c(min(long_psi),3*max(long_psi)-2*min(long_psi)))
  }
}
axis(side=1);axis(side=4);mtext(side=1,line=3,'longitude')
par(new=T)
for (j in 1:r){
  if (j==1){plot(lat_psi[,j],lat_range,col=cl[j],type="l",xaxt='n',yaxt='n',xlab=NA,ylab=NA,
                 ylim=c(min(f[,1])-3,max(f[,1])),xlim=c(min(lat_psi),3*max(lat_psi)-2*min(lat_psi)))}
  else {
    lines(lat_psi[,j],lat_range,col=cl[j],xaxt='n',yaxt='n',xlab=NA,ylab=NA,
          ylim=c(min(f[,1])-3,max(f[,1])),xlim=c(min(lat_psi),3*max(lat_psi)-2*min(lat_psi)))
  }
}
axis(side=2);axis(side=3);mtext(side=2,line=3,'latitude')
legend("topright","(x,y)",1:r,col=cl,lty=rep(1,r))
dev.off()
