function [ MeanLL, StdLL ] = SVMcrossvalMultiLabel_v02( X,yd,C,b  )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
% Partition data into 10 folds for cross-validation

% Train svms successively (with the same features), but decide first
% whether young or old, then apply specialized svm to decide on gender
% and health status

% TODO: Is this a good idea to partition only dependent on the genders?
yc = cvpartition(yd(:,2),'k',10);
% yc = cvpartition(yd,'leaveout');

% Distinguish
idxYoung = yd(:,2) == 1;

NoT = yc.NumTestSets;
SaveLL = zeros(NoT,1);

% Choose kernel
kernel = 'rbf';

% Do cross-validation
for i = 1:NoT
    % Get training and test indices
    trainidx = training(yc,i);
    testidx = test(yc,i);
    trainidxOld = trainidx & ~idxYoung;
    trainidxYoung = trainidx & idxYoung;
    testidxOld = testidx & ~idxYoung;
    testidxYoung = testidx & idxYoung;
    yhat = zeros(sum(testidx),3);
    tempyhat = zeros(sum(testidx),4);
        % Train model
        % Support vector machine
%         model = fitcsvm(X(trainidx,:),yd(trainidx,lcount),'Standardize',true,'KernelFunction','rbf',...
%                 'BoxConstraint',C,'KernelScale',b);
        modelAge = fitcsvm(X(trainidx,:),yd(trainidx,2),'Standardize',true,'KernelFunction',kernel,...
                'BoxConstraint',C,'KernelScale',b);
        modelGenderOld = fitcsvm(X(trainidxOld,:),yd(trainidxOld,1),'Standardize',true,'KernelFunction',kernel,...
                'BoxConstraint',C,'KernelScale',b);
        modelGenderYoung = fitcsvm(X(trainidxYoung,:),yd(trainidxYoung,1),'Standardize',true,'KernelFunction',kernel,...
                'BoxConstraint',C,'KernelScale',b);
        modelHealthOld = fitcsvm(X(trainidxOld,:),yd(trainidxOld,3),'Standardize',true,'KernelFunction',kernel,...
                'BoxConstraint',C,'KernelScale',b);
        modelHealthYoung = fitcsvm(X(trainidxYoung,:),yd(trainidxYoung,3),'Standardize',true,'KernelFunction',kernel,...
                'BoxConstraint',C,'KernelScale',b);
        % Test model with hamming loss
        yhat(:,2) = predict(modelAge,X(testidx,:));
        tempyhat(:,1) = predict(modelGenderOld,X(testidxOld,:));
        tempyhat(:,2) = predict(modelGenderYoung,X(testidxYoung,:));
        tempyhat(:,3) = predict(modelHealthOld,X(testidxOld,:));
        tempyhat(:,4) = predict(modelHealthYoung,X(testidxYoung,:));
        idxYounghat = yhat(:,2) == 1;
        yhat(~idxYounghat,1) = tempyhat(~idxYounghat,1);
        yhat(idxYounghat,1) = tempyhat(idxYounghat,2);
        yhat(~idxYounghat,3) = tempyhat(~idxYounghat,3);
        yhat(idxYounghat,3) = tempyhat(idxYounghat,4);
    compys = sum(xor(yd(testidx,:),yhat),2)*1/3;
    SaveLL(i) = sum(compys)*1/length(yhat);
end

% Compute mean loss and the deviation of the loss
MeanLL = mean(SaveLL);
StdLL = std(SaveLL);

end

