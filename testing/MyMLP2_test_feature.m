addpath('../data','../testing','../ReadData3D_version1k/nii','../feature extract', ...
    '../preprocess');

% Compute histogram bounds
% Tested: 
% split: 6x6x6, bins: 30
% split: 6x6x6, bins: 60
% split: 9x9x9, bins: 30

parameters1 = struct('x_segments',5,'y_segments',5,'z_segments',5);
% parameters1 = struct('xp',0.2,'yp',0.1,'zp',0.2,'levels',2,'dir','z');
% parameters1 = struct('x_segments',5,'y_segments',5,'z_segments',5,...
%     'xp',0.3,'yp',0.3,'zp',0.3,'levels',2,'dir','all');
Xint = generate_X('../data/set_train', 'MLP2_getMinMaxIntensity', parameters1);
hbounds = MLP2_getHistobounds(Xint);
%%

% train features
parameters2 = struct('x_segments',5,'y_segments',5,'z_segments',5,'bins',15,'hbounds',hbounds);
X = generate_X('../data/set_train', 'MLP2_feature_extract3_ar', parameters2);

%%

% % test features -> Check if parameters are equal to the train paras
parameters2 = struct('x_segments',5,'y_segments',5,'z_segments',5,'bins',10,'hbounds',hbounds);
Xtest = generate_X('../data/set_test', 'MLP2_feature_extract3_ar', parameters2);

%%

% Read target data
yd = csvread('../targets.csv');

%%

% Exclude zero data
idxz = sum(X,1) == 0;
X2 = X(:,~idxz);
% Exclude useless features
idxs = std(X2,[],1) == 0;
X3 = X2(:,~idxs);

[m,n] = size(X3);
% Choose 1000 features at random
selecti = randi(n,700,1);
X4 = X3(:,selecti);

X10 = [X3(:,870) X3(:,948)];

%%
X5 = Xtest(:,~idxz);
X6 = X5(:,~idxs);

%%

% idxsick = find(yd == 0);
% idxhealthy = find(yd == 1);
% 
% % Seperate the two classes
% [x,y] = size(X);
% 
% X2 = [X(idxsick,:) ; X(idxhealthy,:)];

% scatter(ax1,y,X(:,1))
%ax2 = subplot(2,1,2);
% scatter(ax2,Y(:,2)./Y(:,1),y,'filled','d')
%scatter(ax2,y,X(:,2),'filled','d')
% 

% % Ensemble optimization
% model = fitcensemble(X,yd,'OptimizeHyperparameters','auto',... 
%     'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
%     'expected-improvement-plus', 'Kfold', 10));
%
% % Support vector machine, gridsearch optimization
% model = fitcsvm(X,yd,'Standardize',true,...
%     'KernelFunction','rbf','ScoreTransform','logit','OptimizeHyperparameters',...
%     'auto','HyperparameterOptimizationOptions',struct('Optimizer','gridsearch','kfold',10));
%
% % Support vector machine, bayes optimization
% model = fitcsvm(X,yd,'Standardize',true,...
%     'KernelFunction','rbf','ScoreTransform','logit','OptimizeHyperparameters',...
%     'auto','HyperparameterOptimizationOptions',struct('kfold',10));

% betarange = [1:9 10:10:90 100:100:500];
% betarange = [100:10:350];
% betarange = [1 80 200];
betarange = 1;
% betarange = [0.1:0.1:0.9 1:9 10];
% Crange = [10:10:90 1e2:1e2:900 1e3:1e3:1e4];
% Crange = linspace(0.01,500,50);
Crange = [1 10 100];
% Crange = [1e3:1e3:1e4];


% lossfun =@(yd,yhat,w)(-mean(yd.*log(yhat(:,2))+(1-yd).*log(1-yhat(:,2)),1));

% SaveCVLLmean = zeros(length(betarange),length(Crange));
% SaveCVLLstd = SaveCVLLmean;

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
tic
disp('Evaluation started...')
parfor i = 1:length(hypcell)
    b = hypcell{i}(1);
    C = hypcell{i}(2);
%   [SaveCVLLmean(ib,ic), SaveCVLLstd(ib,ic)] = SVMcrossval(X3,yd,C,b);
    [SaveCVLLcm{i}, SaveCVLLcs{i}] = SVMcrossval(X10,yd,C,b);
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
k = 3;
figure(1)
plot(Crange,SaveCVLLmean(k,:))
figure(2)
plot(Crange,SaveCVLLstd(k,:))
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
figure(3)

scatter(X3(idxsick,870),X3(idxsick,948),'r')
hold on
scatter(X3(~idxsick,870),X3(~idxsick,948),'b')
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

% C = find(Crange == 4000);
% 
% figure(2)
% plot(betarange,SaveLL(:,C))


% CV method: resubstitution
% first attempt: beta = 200, C = 4000;
%
% CV method: 10 k-fold -> used labels instead of scores to evaluate LL
% second attempt: - corrupt beacause of mistake in SVMcrossval
% Min 1: beta = 300, C = 4000, MeanLL = 2.275, StdLL = 0.7743;
% Min 2: beta = 400, C = 400, MeanLL = 2.278, StdLL = 0.5635
% Min 3: beta = 40, C = 400, MeanLL = 2.277, StdLL = 0.5189
%
% third attempt: Excluded zero data - corrupt
% Min 1: beta = 10, C = 400, MeanLL = 2.773, StdLL = 0.1710;
% Min 2: bera = 10, C = 1e4, MeanLL = 2.773, StdLL = 0.1710;
% Min 3: beta = 100, C = 400, MeanLL = 2.774, StdLL = 0.8331
%
% fourth attempt: Exluded zero and useless feature - corrupt
% Min 1: beta = 60, C = 400, MeanLL = 2.277, StdLL = 
% Min 2: beta = 10, C = 1e4, MeanLL = 2.237, StdLL = 



%%
% Find beta and C index
bidx = find(betarange == round(10^1.778));
Cidx = find(Crange == round(10^3));

%%

% Train model
% C = 1000;
% b = 60;
C = 10;
b = 90;
[CVm, CVstd] = SVMcrossval(X3,yd,C,b);
model = fitcsvm(X3,yd,'Standardize',true,'KernelFunction','rbf',...
            'BoxConstraint',C,'KernelScale',b,'ScoreTransform','doublelogit');
[~,yresub] = predict(model,X3);
idx0 = yresub == 0;
yresub(idx0) = 0.00001;
idx1 = yresub == 1;
yresub(idx1) = 0.99999;
    
LLresub = -mean(yd.*log(yresub(:,2))+(1-yd).*log(1-yresub(:,2)),1);
% LLresub = -mean(yd.*log(yresub)+(1-yd).*log(1-yresub),1);

[~,yhat] = predict(model,X6);


%%

% Test naive bayes
% [SaveCVLLmean, SaveCVLLstd] = NBcrossval(X4(:,2:end),yd);
% 
% %%
% 
% % % Test if there is intraclass variance
% 
% idxc = yd == 0;
% idxicvs = std(X3(idxc,:),[],1) == 0;
% idxicvh = std(X3(~idxc,:),[],1) == 0;
% 
% idxch = idxicvs | idxicvh;
% 
% X4 = X3(:,~idxch);



