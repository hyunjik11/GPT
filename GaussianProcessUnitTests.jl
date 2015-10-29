using GaussianProcess
import PyPlot.figure
srand(17);
f=SECov(0.5,1);
m=x->norm(x)^2;

function unitTest1d(gp::GP,i::Integer)
    println(GPmean(gp))
    println(GPcov(gp))
    gp_post=GPpost(gp,[-0.5,0,0.5],[1,2,1],0.1);
    println(GPrand(gp_post,[-0.1,0,0.1]))
    figure(i)
    plot1d(gp_post)
    return nothing
end

function unitTest2d(gp::GP)
    println(GPmean(gp))
    println(GPcov(gp))
    gp_post=GPpost(gp,[1 2;3 4],[1,2],0.1);
    println(GPrand(gp_post,[2 1;4 3]))
    return nothing
end

gp=GP(0,f,1);
unitTest1d(gp,1)

gp=GP(m,f,1);
unitTest1d(gp,2)

gp=GP(0,f,2);
unitTest2d(gp)

gp=GP(m,f,2);
unitTest2d(gp)
