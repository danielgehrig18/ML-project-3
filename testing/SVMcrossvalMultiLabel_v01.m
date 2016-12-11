function [ MeanLL, StdLL ] = SVMcrossvalMultiLabel_v01( X,yd,C,b,lcount  )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
% Partition data into 10 folds for cross-validation

% Train each svm independently, with the same features

% TODO: Is this a good idea to partition only dependent on the genders?
yc = cvpartition(yd(:,lcount),'k',10);
% yc = cvpartition(yd,'leaveout');

kernel = 'rbf';

NoT = yc.NumTestSets;
SaveLL = zeros(NoT,1);

% Do cross-validation
for i = 1:NoT
    % Get training and test indices
    trainidx = training(yc,i);
    testidx = test(yc,i);
        % Train model
        % Support vector machine
%         model = fitcsvm(X(trainidx,:),yd(trainidx,lcount),'Standardize',true,'KernelFunction','rbf',...
%                 'BoxConstraint',C,'KernelScale',b);
        model = fitcsvm(X(trainidx,:),yd(trainidx,lcount),'Standardize',true,'KernelFunction',kernel,...
                'BoxConstraint',C,'KernelScale',b);
        % Test model with hamming loss
        yhat = predict(model,X(testidx,:));
    xorys = xor(yd(testidx,lcount),yhat);
    SaveLL(i) = sum(xorys)*1/length(yhat);
end

% Compute mean loss and the deviation of the loss
MeanLL = mean(SaveLL);
StdLL = std(SaveLL);

end

