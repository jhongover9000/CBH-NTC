%% Testing plotting a series of SHAP - 2 class
% clear all
% clc
SHAP_0=readNPY('SHAPS_0.npy');
SHAP_1=readNPY('SHAPS_1.npy');
load('EEG_chlocs.mat');
timepoints = 768;



figure(1)
subplot(2,1,1)
contourf(1:1:timepoints,1:1:56,rescale(SHAP_0),40,'linecolor','none')
xlim([1 50])
subplot(2,1,2)
contourf(1:1:timepoints,1:1:56,rescale(SHAP_1),40,'linecolor','none')
xlim([1 50])

%% continue

t0=1;
n=4;
step = timepoints/n;
figure(2)
SHAP_0=SHAP_0./(sum(sum(SHAP_0)));
SHAP_1=SHAP_1./(sum(sum(SHAP_1)));
SHAP=(SHAP_0+SHAP_1)/2;
z1=0;
z2=1e-5;

z11=0;
z21=1e-5;
c=colormap("jet");


for i=1:n
    subplot(2,n,i)
    temp1 = mean(abs(SHAP_0(:,t0+((i-1)*step):t0+(i*step)-1)),2);
    topoplot(temp1, chanlocsEasyCapNoRef,'maplimits',[z1 z2],'electrodes','off','colormap',c,'style','map' ) %select a condition
% 
    subplot(2,n,i+n)  
    temp2 = mean(abs(SHAP_1(:,t0+((i-1)*step):t0+(i*step)-1)),2);
    topoplot(temp2, chanlocsEasyCapNoRef,'maplimits',[z1 z2],'electrodes','off','colormap',c,'style','map' ) %select a condition


%     subplot(2,n,i+n+n)  
% 
%     temp2 = mean(abs(SHAP(:,t0+((i-1)*step):t0+(i*step)-1)),2);
%     topoplot(temp2, chanlocsEasyCapNoRef,'maplimits',[z1 z2],'electrodes','off','colormap',c,'style','map' ) %select a condition

end

set(gcf,'Position',[300 500 900 480])
