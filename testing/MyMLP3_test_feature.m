addpath('../data','../testing','../ReadData3D_version1k/nii','../feature extract', ...
    '../preprocess');

% %% Filter images
% parametersAge = struct('boxSize',[7 7 7]);
% generate_X('../data/set_train', 'MLP3_filterImages_ar', parametersAge);

%% train features, this feature showed quite good performance for age classification
parametersAge = struct('x_segments',3,'y_segments',3,'z_segments',3,'bins',10,'redIm',[0.4 0.4 0.4],'filterOn',false);
XAge = generate_X('../data/set_train', 'MLP3_feature_extract3_ar', parametersAge);

%% train new features for gender and health status
filterOn = false;
if filterOn
    path = '../data/set_train_filt';
else
    path = '../data/set_train';
end
parametersGender = struct('x_segments',6,'y_segments',6,'z_segments',6,'bins'...
    ,10,'redIm',[0.2 0.2 0.2],'filterOn',filterOn,'imAdjustOn',true);
XGender = generate_X(path,'MLP3_feature_extract3_ar', parametersGender);

%%

% % test features -> Check if parameters are equal to the train paras
parameters2 = struct('x_segments',5,'y_segments',5,'z_segments',5,'bins',10,'hbounds',hbounds);
Xtest = generate_X('../data/set_test', 'MLP2_feature_extract3_ar', parameters2);

%%

% Read target data
ydTot = csvread('../targets.csv');
youngIdx = ydTot(:,2) == 1;

% Train on young people only
% yd = ydTot(youngIdx,:);
yd = ydTot;

% ycol = yd(:,3);

%% Choose only the third bin of each chunk
select3rd = 3:5:40;

X3 = XGender(:,select3rd);

%% Exclude zero data
idxz = sum(X,1) == 0;
X2 = X(:,~idxz);
% Exclude useless features
idxs = std(X2,[],1) == 0;
X3 = X2(:,~idxs);

%%

[m,n] = size(X3);
% Choose 1000 features at random
selecti = randi(n,700,1);
X4 = X3(:,selecti);

X10 = [X3(:,870) X3(:,948)];

%%
X5 = Xtest(:,~idxz);
X6 = X5(:,~idxs);

%%

% betarange = 10;
% betarange = [100:10:350];
betarange = [1 80 200];
% betarange = 1;
% betarange = [0.1:0.1:0.9 1:9 10];
Crange = [10:10:90 1e2:1e2:900 1e3:1e3:1e4];
% Crange = linspace(0.01,500,50);
% Crange = [1 10 100];
% Crange = [1e3:1e3:1e4];

SaveCVLLcm = cell(length(betarange)*length(Crange),1);
SaveCVLLcs = SaveCVLLcm;

% Create hyperparameter cell
hypcell = cell(length(betarange)*length(Crange),1);
count = 0;
for ib = 1:length(betarange)
    for ic = 1:length(Crange)
        count = count+1;
        hypcell{count} = [betarange(ib) Crange(ic)];
    end
end

% h = waitbar(0,'Evaluating...');
% count = 0;
% totc = length(betarange)*length(Crange);
lcount = 3;

tic
disp('Evaluation started...')
for i = 1:length(hypcell)
    b = hypcell{i}(1);
    C = hypcell{i}(2);
%   [SaveCVLLmean(ib,ic), SaveCVLLstd(ib,ic)] = SVMcrossval(X3,yd,C,b);
    [SaveCVLLcm{i}, SaveCVLLcs{i}] = SVMcrossvalMultiLabel_v01(XGender,yd,C,b,lcount);
%   count = count+1;
%   waitbar(count/totc);
end
% close(h)
t = toc;
disp('Evaluation completed')

SaveCVLLmean_ = cell2mat(SaveCVLLcm);
SaveCVLLstd_ = cell2mat(SaveCVLLcs);
SaveCVLLmean__ = reshape(SaveCVLLmean_,length(Crange),length(betarange));
SaveCVLLstd__ = reshape(SaveCVLLstd_,length(Crange),length(betarange));
SaveCVLLmean = SaveCVLLmean__';
SaveCVLLstd = SaveCVLLstd__';

% 
% 
%%
% Plot with fixed C
close all
k = 3;
figure(1)
semilogx(Crange,SaveCVLLmean)
legend('1','2','3')
figure(2)
semilogx(Crange,SaveCVLLstd)
%%

idxsick = yd == 0;
meansick = mean(X3(idxsick,:),1);
meanheal = mean(X3(~idxsick,:),1);
meandiff = abs(meansick-meanheal);
stdsick = std(X3(idxsick,:),1);
stdheal = std(X3(~idxsick,:),1);
stdsum = stdsick+stdheal;

figure(4)
plot(1:n,meandiff,'b')
hold on
plot(1:n,stdsum,'r')
hold off
idximp = meandiff > 200;

%%
idxmale = yd(:,3) == 0;
figure(3)
scatter(XGender(idxmale,13),XGender(idxmale,28),'r')
hold on
scatter(XGender(~idxmale,13),XGender(~idxmale,28),'b')
hold off
%%
% 
% [~,yhat] = predict(model,X);
% % 
% % %%
% % 
% % tscore = 1./(1+exp(-1*yhat(:,2)));
% % %%
% % 
% LL = -mean(yd.*log(yhat(:,2))+(1-yd).*log(1-yhat(:,2)),1);

% Plot SaveLL on log10 scale
close all

figure(1)
mesh(log10(Crange),log10(betarange),SaveCVLLmean)
title('Mean cross-validation loss')
ylabel('\beta')
xlabel('C')
zlabel('Binary Log-Loss')


figure(2)
mesh(log10(Crange),log10(betarange),SaveCVLLstd)
title('Deviation of the cross-validation loss')
ylabel('\beta')
xlabel('C')
zlabel('Binary Log-Loss')

figure(3)
pcolor(log10(Crange),log10(betarange),SaveCVLLmean)
title('Mean cross-validation loss')
ylabel('\beta')
xlabel('C')

figure(4)
pcolor(log10(Crange),log10(betarange),SaveCVLLstd)
title('Deviation of the cross-validation loss')
ylabel('\beta')
xlabel('C')



