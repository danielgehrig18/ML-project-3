global model;

%% get features
% choose function and its parameters
fun = 'MLP2_feature_extract3'; % TODO: modify function
parameters = struct('x_segments', s, ... % TODO: optimize parameters through CV
                    'y_segments', s, ...
                    'z_segments', s, ...
                    'bins',b);

X = generate_X('data/set_train', fun, parameters);

y = csvread('targets.csv');

%% train model
model = fitcensemble(X,y,'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
    'expected-improvement-plus', 'MaxObjectiveEvaluations', 30, 'SaveIntermediateResults', 0, 'Verbose', 1, 'ShowPlots', 1, 'Kfold', 20));


%% crossvalidation
cv_model = crossval(model); % TODO: way to pass model function and parameters

CV = kfoldfun(cv_model, @crossvalidation);
m = mean(CV);
std = mean((CV-m).^2)^.5;

disp(['mean: ' num2str(m) 'std: ' num2str(std)]); 
