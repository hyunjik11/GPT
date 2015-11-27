c=h5open("theta.h5","w") do file
    write(file,"theta_store",theta_store)
    write(file,"theta_store2",theta_store2)
end

file="theta.h5";
theta_store=h5read(file,"theta_store");
theta_store2=h5read(file,"theta_store2");

figure()
for i=1:10
    subplot(2,5,i)
    ax=gca()
    plot(vec(theta_store[10*i,:]))
    plot(vec(theta_store2[10*i,:]))
    ylim(-5,5)
    xlabel("num_iterations(no burnin)")
    setp(ax[:get_xticklabels](),fontsize=8)
end

savefig("/homes/hkim/GPT/Plots/NoTensorSGLDThetaTrace")
