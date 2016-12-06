function [ betas, statistics] = train_b_cv_v3( x_folder, y_file, fun, parameters, NoF, model)
%TRAIN_B Trains a b parameter vector for linear regression with data from
%the x_folder and targets from the y_file. limit1 and limit2 are parameters
%for the feature_extract function.
% Only compatible with optimize_feature_ar
% New: x2fx function -> additional input argument, and lasso optimization

% load targets
y = csvread(y_file);

% generate #data_points x (#features+1) data matrix
X = generate_X_optver(x_folder, fun, parameters, NoF,'train');

% Convert predictors to design matrix
Xdes = x2fx(X,model);

% Do LASSO regression, use 10 fold cross-validation
[betas, statistics] = lasso(Xdes,y,'CV',10);


end

