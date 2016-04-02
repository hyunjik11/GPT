figure();
subplot(2,1,1);
hold on
m_values=[1,2,3,4,5,6];
plot(m_values,half_logdet*ones(size(m_values)));
errorbar(m_values,ld_rff,std_ld_rff);
errorbar(m_values,naive_ld_means,naive_ld_stds);
errorbar(m_values,fic_ld_means,fic_ld_stds);
errorbar(m_values,pic_ld_means,pic_ld_stds);
set(gca,'XTick',[1 2 3 4 5 6]);
set(gca,'XTickLabel',[10 20 40 80 160 320]);
legend('Exact GP','RFF','DTC','FIC','PIC')
ylabel('logdet/2')
xlabel('m')
hold off

subplot(2,1,2);
hold on
m_values=[1,2,3,4,5,6];
plot(m_values,half_innerprod*ones(size(m_values)));
errorbar(m_values,innerprod_rff,std_innerprod_rff);
errorbar(m_values,naive_ip_means,naive_ip_stds);
errorbar(m_values,fic_ip_means,fic_ip_stds);
errorbar(m_values,pic_ip_means,pic_ip_stds);
set(gca,'XTick',[1 2 3 4 5 6]);
set(gca,'XTickLabel',[10 20 40 80 160 320]);
legend('Exact GP','RFF','DTC','FIC','PIC')
ylabel('innerprod/2')
xlabel('m')
hold off