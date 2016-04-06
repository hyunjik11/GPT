figure();
subplot(2,2,1);
hold on
m_values=[1,2,3,4,5,6];
plot(m_values,half_logdet*ones(size(m_values)));
errorbar(m_values,ld_rff,std_ld_rff);
errorbar(m_values,naive_ld_means,naive_ld_stds);
errorbar(m_values,fic_ld_means,fic_ld_stds);
errorbar(m_values,pic_ld_means,pic_ld_stds);
set(gca,'XTick',[1 2 3 4 5 6]);
set(gca,'XTickLabel',[100 200 400 800 1600 3200]);
legend('Exact GP','RFF','DTC','FIC','PIC')
ylabel('logdet/2')
xlabel('m')
hold off

subplot(2,2,2);
hold on
m_values=[1,2,3,4,5,6];
plot(m_values,half_innerprod*ones(size(m_values)));
errorbar(m_values,innerprod_rff,std_innerprod_rff);
errorbar(m_values,naive_ip_means,naive_ip_stds);
errorbar(m_values,fic_ip_means,fic_ip_stds);
errorbar(m_values,pic_ip_means,pic_ip_stds);
set(gca,'XTick',[1 2 3 4 5 6]);
set(gca,'XTickLabel',[100 200 400 800 1600 3200]);
ylim([3000 7000])
legend('Exact GP','RFF','DTC','FIC','PIC')
ylabel('innerprod/2')
xlabel('m')
hold off

subplot(2,2,3);
hold on
m_values=[1,2,3,4,5,6];
plot(m_values,frob_svd);
errorbar(m_values,mean_frob_rff,std_frob_rff);
errorbar(m_values,mean_frob_naive,std_frob_naive);
errorbar(m_values,mean_frob_fic,std_frob_fic);
errorbar(m_values,mean_frob_pic,std_frob_pic);
set(gca,'XTick',[1 2 3 4 5 6]);
set(gca,'XTickLabel',[100 200 400 800 1600 3200]);
legend('SVD','RFF','DTC','FIC','PIC')
ylabel('Frobenius Norm Error')
xlabel('m')
hold off

subplot(2,2,4);
hold on
m_values=[1,2,3,4,5,6];
plot(m_values,spec_svd);
errorbar(m_values,mean_spec_rff,std_spec_rff);
errorbar(m_values,mean_spec_naive,std_spec_naive);
errorbar(m_values,mean_spec_fic,std_spec_fic);
errorbar(m_values,mean_spec_pic,std_spec_pic);
set(gca,'XTick',[1 2 3 4 5 6]);
set(gca,'XTickLabel',[100 200 400 800 1600 3200]);
legend('SVD','RFF','DTC','FIC','PIC')
ylabel('Spectral Norm Error')
xlabel('m')
hold off