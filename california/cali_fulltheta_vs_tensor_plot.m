figure();
subplot(1,4,1);
n_values=[1,2,3,4,5,6];
hold on
plot(n_values,gptrain*ones(length(n_values),1));
plot(n_values,fullthetatrain);
ylim([0.15 0.50]);
legend('gp-train','fulltheta-train');
set(gca,'XTick',[1 2 3 4 5 6]);
set(gca,'XTickLabel',[100 200 400 800 1600 3200]);
xlabel('n')
ylabel('RMSE')
hold off

subplot(1,4,2);
n_values=[1,2,3,4];
hold on
plot(n_values,gptrain*ones(length(n_values),1)); 
plot(n_values,tensor2train);
plot(n_values,tensor5train);
plot(n_values,tensor10train);
ylim([0.15 0.50]);
legend('gp-train','tensor-train-r=2','tensor-train-r=5','tensor-train-r=10');
set(gca,'XTick',[1 2 3 4]);
set(gca,'XTickLabel',[25 50 100 200]);
xlabel('n')
ylabel('RMSE')
hold off

subplot(1,4,3);
n_values=[1,2,3,4,5,6];
hold on
plot(n_values,gptest*ones(length(n_values),1));
plot(n_values,fullthetatest);
ylim([0.15 0.50]);
legend('gp-test','fulltheta-test');
set(gca,'XTick',[1 2 3 4 5 6]);
set(gca,'XTickLabel',[100 200 400 800 1600 3200]);
xlabel('n')
ylabel('RMSE')
hold off

subplot(1,4,4);
n_values=[1,2,3,4];
hold on
plot(n_values,gptest*ones(length(n_values),1)); 
plot(n_values,tensor2test);
plot(n_values,tensor5test);
plot(n_values,tensor10test);
ylim([0.15 0.50]);
legend('gp-test','tensor-test-r=2','tensor-test-r=5','tensor-test-r=10');
set(gca,'XTick',[1 2 3 4]);
set(gca,'XTickLabel',[25 50 100 200]);
xlabel('n')
ylabel('RMSE')
hold off
