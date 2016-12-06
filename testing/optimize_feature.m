diary;
samples = [];
hold on;

% test feature

cgone = 220;
cgtwo = 750;
cgthree = 1450;
rangeone = 50;
rangetwo = 35;
rangethree = 110;

w_mults = 1:.25:6;

fun = 'feature_extract_3peaks';

RMSE = [];
y = [];
% iterate through parameters and display results
disp(strcat('STARTED OPTIMIZATION OF "', fun, '".')); 
for w_mult=w_mults

    % define parameter struct
    parameters = struct('cgone', 220, ...
                 'cgtwo', 750, ...
                 'cgthree', 1450, ...
                 'rangeone', 50, ...
                 'rangetwo', 35, ...
                 'rangethree', 110, ...
                 'w_mult', w_mult);


    % train the model with the parameters and function
    [model, X] = train_b('data/set_train', 'data/targets.csv', fun, parameters);

    % save sample in data structure
    RMSE = [RMSE, struct('RMSE', model.RMSE, ...
                         'Parameters', parameters, ...
                         'BIC', model.ModelCriterion.BIC, ...
                         'Rsquared', model.Rsquared)];

    % display information
    disp('------------------');
    disp('Parameters: ');
    disp(parameters);
    disp('RMSE: ');
    disp(model.RMSE);
    disp('R values: '); 
    disp(model.Rsquared);
    disp('Bayesian Information Criterion: ');
    disp(model.ModelCriterion.BIC);
    y = [y, model.ModelCriterion.BIC];
    plot(y);
    pause(0.5)
end


