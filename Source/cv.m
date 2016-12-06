function [ criterion ] = cv(XTRAIN,ytrain,XTEST,ytest)
%CV Summary of this function goes here
%   Detailed explanation goes here

model = fitcsvm(XTRAIN, ytrain, 'Standardize',true, 'ScoreTransform', 'logit', 'Prior', [50, 228]/278);
[~,s] = predict(model, XTEST);
yfit = s(:,2);

criterion = -mean(ytest .* log(yfit) + (1 - ytest) .* log(1 - yfit));

end

