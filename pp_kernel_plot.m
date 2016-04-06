figure();
subplot(2,1,1);
hold on
m_values=[1,2,3,4,5,6];
plot(m_values,frob_svd);
errorbar(m_values,mean_frob_rff,std_frob_rff);
errorbar(m_values,mean_frob_naive,std_frob_naive);
errorbar(m_values,mean_frob_fic,std_frob_fic);
errorbar(m_values,mean_frob_pic,std_frob_pic);
set(gca,'XTick',[1 2 3 4 5 6]);
set(gca,'XTickLabel',[10 20 40 80 160 320]);
ylim([0 5])
legend('SVD','RFF','DTC','FIC','PIC')
ylabel('Frobenius Norm Error')
xlabel('m')
hold off

subplot(2,1,2);
hold on
m_values=[1,2,3,4,5,6];
plot(m_values,spec_svd);
errorbar(m_values,mean_spec_rff,std_spec_rff);
errorbar(m_values,mean_spec_naive,std_spec_naive);
errorbar(m_values,mean_spec_fic,std_spec_fic);
errorbar(m_values,mean_spec_pic,std_spec_pic);
set(gca,'XTick',[1 2 3 4 5 6]);
set(gca,'XTickLabel',[10 20 40 80 160 320]);
ylim([0 5])
legend('SVD','RFF','DTC','FIC','PIC')
ylabel('Spectral Norm Error')
xlabel('m')
hold off