
clear ; clc; close all
gt1 = (csvread('c:\users\mohamadmahdi\desktop\GT1.csv'));gt1=gt1(2:end,:);
gt2 = (csvread('c:\users\mohamadmahdi\desktop\GT2.csv'));gt2=gt2(2:end,:);
TR1_JPDA = (csvread('c:\users\mohamadmahdi\desktop\TR1_JPDA.csv'));TR1_JPDA=TR1_JPDA(2:end,:);
TR2_JPDA = (csvread('c:\users\mohamadmahdi\desktop\TR2_JPDA.csv'));TR2_JPDA=TR2_JPDA(2:end,:);
TR1_PDA = (csvread('c:\users\mohamadmahdi\desktop\TR1_PDA.csv'));TR1_PDA=TR1_PDA(2:end,:);
TR2_PDA = (csvread('c:\users\mohamadmahdi\desktop\TR2_PDA.csv'));TR2_PDA=TR2_PDA(2:end,:);

real1 = (csvread('c:\users\mohamadmahdi\desktop\REAL1.csv'));real1=real1(2:end,:);
real2 = (csvread('c:\users\mohamadmahdi\desktop\REAL2.csv'));real2=real2(2:end,:);

clutter = (csvread('c:\users\mohamadmahdi\desktop\CLUTTERs.csv'));clutter=clutter(2:end,:);

T1_gnn = (csvread('c:\users\mohamadmahdi\desktop\T1_gnn.csv'));T1_gnn=T1_gnn(2:end,:);
T2_gnn = (csvread('c:\users\mohamadmahdi\desktop\T2_gnn.csv'));T2_gnn=T2_gnn(2:end,:);

w1_pda = (csvread('c:\users\mohamadmahdi\desktop\weight1_pda.csv'));w1_pda=w1_pda(2:end,:);
w2_pda = (csvread('c:\users\mohamadmahdi\desktop\weight2_pda.csv'));w2_pda=w2_pda(2:end,:);

w1_jpda = (csvread('c:\users\mohamadmahdi\desktop\weight1_jpda.csv'));w1_jpda=w1_jpda(2:end,:);
w2_jpda = (csvread('c:\users\mohamadmahdi\desktop\weight2_jpda.csv'));w2_jpda=w2_jpda(2:end,:);



%%
figure(1)
plot(gt1(1,:), gt1(3,:),'g--') ; hold on;
plot(gt2(1,:), gt2(3,:),'r--') ; hold on;

plot(TR1_JPDA(1,:), TR1_JPDA(3,:),'g') ; hold on;
plot(TR2_JPDA(1,:), TR2_JPDA(3,:),'r') ; hold on;

scatter(real1(1,:) ,real1(2,:),'ko') ;hold on
scatter(real2(1,:) ,real2(2,:),'ko') ;hold on

scatter(clutter(1,:) ,clutter(2,:),'y^') ;hold on
title('JPDA')
legend('ground truth1','ground truth2','track 1 JPDA','track 2 JPDA','sensor','','clutter')

%%
figure(2)
plot(gt1(1,:), gt1(3,:),'g--') ; hold on;
plot(gt2(1,:), gt2(3,:),'r--') ; hold on;

plot(TR1_PDA(1,:), TR1_PDA(3,:),'g') ; hold on;
plot(TR2_PDA(1,:), TR2_PDA(3,:),'r') ; hold on;
title('PDA')
legend('ground truth1','ground truth2','track 1 PDA','track 2 PDA')

%%
figure(3)
plot(gt1(1,:), gt1(3,:),'g--') ; hold on;
plot(gt2(1,:), gt2(3,:),'r--') ; hold on;

plot(T1_gnn(1,:), T1_gnn(3,:),'g') ; hold on;
plot(T2_gnn(1,:), T2_gnn(3,:),'r') ; hold on;
title('GNN')
legend('ground truth1','ground truth2','track 1 PDA','track 2 PDA')
