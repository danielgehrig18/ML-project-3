function [ m, MSE] = train_b_cv_v4( x_folder, y_file, fun, parameters, NoF)
%TRAIN_B Trains a b parameter vector for linear regression with data from
%the x_folder and targets from the y_file. limit1 and limit2 are parameters
%for the feature_extract function.
% Only compatible with optimize_feature_ar
% New: Use stepwise linear regression method and quartic terms

% load targets
y = csvread(y_file);

% generate #data_points x (#features+1) data matrix
X = generate_X_optver(x_folder, fun, parameters, NoF,'train');

% Convert predictors to design matrix
% Xdes = x2fx(X,model);
Xdes = [X X.^2 X.^3 X.^4];

% Do LASSO regression, use 10 fold cross-validation
m = stepwiseglm(Xdes, y, 'interactions', 'Criterion', 'bic', 'Exclude', [], 'Distribution', 'inverse gaussian');

y_hat = predict(m,Xdes);
MSE = sum(abs(y-y_hat).^2);

end

