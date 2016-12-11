function [ MeanLL, StdLL ] = SVMcrossval( X,yd,C,b  )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
% Partition data into 10 folds for cross-validation
yc = cvpartition(yd,'k',10);
% yc = cvpartition(yd,'leaveout');

NoT = yc.NumTestSets;
SaveLL = zeros(NoT,1);
sigmoid =@(x,is)(1./(1+exp(-is*x)));

% Do cross-validation
for i = 1:NoT
    % Get training and test indices
    trainidx = training(yc,i);
    testidx = test(yc,i);
    % Train model
    % Support vector machine
    model = fitcsvm(X(trainidx,:),yd(trainidx),'Standardize',true,'KernelFunction','rbf',...
            'BoxConstraint',C,'KernelScale',b);
%   ,'ScoreTransform','doublelogit'
    % Test model, with cross entropy loss
    [~,yhat] = predict(model,X(testidx,:));
    yhat = yhat(:,2);
    yhat = sigmoid(yhat,2);
    % % find yhat = 0 and = 1, to prevent NaN entries
    idx0 = yhat == 0;
    yhat(idx0) = 0.00001;
    idx1 = yhat == 1;
    yhat(idx1) = 0.99999;
    % Compute loss
    SaveLL(i) = -mean(yd(testidx).*log(yhat)+(1-yd(testidx)).*log(1-yhat),1);
%     % Test model with hamming loss
%     yhat = predict(model,X(testidx,:));
%     compys = xor(yd(testidx),yhat);
%     SaveLL(i) = sum(compys)*1/length(yhat);
end

% Compute mean loss and the deviation of the loss
MeanLL = mean(SaveLL);
StdLL = std(SaveLL);

end

