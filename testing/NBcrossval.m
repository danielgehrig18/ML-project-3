function [ MeanLL, StdLL ] = NBcrossval( X,yd  )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
% Partition data into 10 folds for cross-validation
% yc = cvpartition(yd,'k',10);
% yc = cvpartition(yd,'leaveout');
% NoT = yc.NumTestSets;

% Bypass cross validation (do resubstitution)
yc = yd;
NoT = 1;

SaveLL = zeros(NoT,1);

% Do cross-validation
for i = 1:NoT
    % Get training and test indices
%     trainidx = training(yc,i);
%     testidx = test(yc,i);
    trainidx = ones(length(yc),1);
    testidx = trainidx;
    % Train model
    % Naive Bayes classifier
    model = fitcnb(X(trainidx,:),yd(trainidx));
    % Test model
    [~,yhat] = predict(model,X(testidx,:));
    yhat = yhat(:,2);
    % % find yhat = 0 and = 1, to prevent NaN entries
    idx0 = yhat == 0;
    yhat(idx0) = 0.00001;
    idx1 = yhat == 1;
    yhat(idx1) = 0.99999;
    % Compute loss
    SaveLL(i) = -mean(yd(testidx).*log(yhat)+(1-yd(testidx)).*log(1-yhat),1);
end

% Compute mean loss and the deviation of the loss
MeanLL = mean(SaveLL);
StdLL = std(SaveLL);

end

