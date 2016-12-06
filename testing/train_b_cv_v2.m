function [ betas, X, RMSE, cvRMSE] = train_b_cv_v2( x_folder, y_file, fun, parameters, NoF )
%TRAIN_B Trains a b parameter vector for linear regression with data from
%the x_folder and targets from the y_file. limit1 and limit2 are parameters
%for the feature_extract function.
% Only compatible with optimize_feature_ar_fe05 and get_CSF_Tissue_03

% load targets
y = csvread(y_file);

% generate #data_points x (#features+1) data matrix
X = generate_X_optver(x_folder, fun, parameters, NoF,'train');

% create linear model
% model = LinearModel.fit(X, y, 'RobustOpts', 'off');

% Do cross-validation
% Add ones column to X matrix (intercept)
XincOnes = [ones(length(y),1),X];
% Choose partition style
% Number of observed data
n = size(XincOnes,1);
% Data part hold out for cross-validation
p = 0.15;
c = cvpartition(n,'HoldOut',p);

% Function for multilinear regression needed in crossval
% regf =@(XTRAIN,ytrain,XTEST)(XTEST*regressfun(ytrain,XTRAIN));

regfun =@ regf;

% Cross-validation, output MSE
cvMSE = crossval('mse',XincOnes,y,'Predfun',regfun,'partition',c);
cvRMSE = sqrt(cvMSE);

% Compute beta vector (estimator), simple multi-linear regression
betas = regressfun(y,XincOnes);
% Compute Root mean squared error
RMSE = sqrt(mean((y-regf(XincOnes,y,[])).^2));

end

function[y_hat] = regf(XTRAIN,ytrain,XTEST)
% Compute y_hat (average)
    betas = regressfun(ytrain,XTRAIN);
    
    if isempty(XTEST)
        % use to compute y_hat of given dataset (not for cv)
        XTEST = XTRAIN;
    end
    
    [m,~] = size(XTEST);
    
    X1 = [ones(m,1) XTEST(:,2)];
    X2 = [ones(m,1) XTEST(:,3)];
    X3 = [ones(m,1) XTEST(:,4)];
    
    y = zeros(m,3);
    
    y(:,1) = X1*betas(:,1);
    y(:,2) = X2*betas(:,2);
    y(:,3) = X3*betas(:,3);
    
    y_hat = mean(y,2);
      
end

function [ bs ] = regressfun(y,XincOnes)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
    [m,~] = size(XincOnes);
    
    Xf1 = [ones(m,1) XincOnes(:,2)];
    Xf2 = [ones(m,1) XincOnes(:,3)];
    Xf3 = [ones(m,1) XincOnes(:,4)];
    
    b1 = regress(y,Xf1);
    b2 = regress(y,Xf2);
    b3 = regress(y,Xf3);
    
    bs = [b1, b2, b3];

end


