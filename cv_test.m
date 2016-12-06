clear all;
clc;

global model;

%% get features
% choose function and its parameters
fun = 'MLP2_feature_extract3'; % TODO: modify function
parameters = struct('x_segments', 3, ... % TODO: optimize parameters through CV
                    'y_segments', 3, ...
                    'z_segments', 3, ...
                    'bins',3);

X = generate_X('data/set_train', fun, parameters);

y = csvread('targets.csv');

%% train model
model1 = fitcsvm(X,y(:,1),'OptimizeHyperparameters','all',...
    'HyperparameterOptimizationOptions',struct('Optimizer', 'gridsearch', 'AcquisitionFunctionName',...
    'expected-improvement-plus'));
model2 = fitcsvm(X,y(:,2),'OptimizeHyperparameters','all',...
    'HyperparameterOptimizationOptions',struct('Optimizer', 'gridsearch', 'AcquisitionFunctionName',...
    'expected-improvement-plus'));
model3 = fitcsvm(X,y(:,3),'OptimizeHyperparameters','all',...
    'HyperparameterOptimizationOptions',struct('Optimizer', 'gridsearch', 'AcquisitionFunctionName',...
    'expected-improvement-plus'));


%% crossvalidation
cv_model1 = crossval(model1); % TODO: way to pass model function and parameters
cv_model2 = crossval(model2); % TODO: way to pass model function and parameters
cv_model3 = crossval(model3); % TODO: way to pass model function and parameters

CV = kfoldfun(cv_model, @crossvalidation);
m = mean(CV);
std = mean((CV-m).^2)^.5;

disp(['mean: ' num2str(m) 'std: ' num2str(std)]); 
