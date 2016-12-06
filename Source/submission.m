function [y_hat] = submission( folder, file, model,  fun, parameters)
%   SUBMISSION Calculates the expected target values with b for test data and
%   writes it into file

%   Args:   folder:     folder with all the test data for X
%           file:       file path where the submission file is written
%           model:      model containing the parameters to construct the predictions with the test data matrix. 
%           fun:        function to be used for the feature extraction
%           parameters: struct containing all relevant arguments to execute
%                       fun
%
%   Return: y_hat:     predicted y values 

% generate test data matrix. Has dimensions 
% #test_data_points x (#features + 1)
X = generate_X(folder, fun, parameters);

% calculate the test targets
[~,y_hat] = predict(model, X); 
y_hat = y_hat(:,2);

%% check if there is already a file with name 'submit.csv', if so delete it
if exist(file, 'file') == 2
    delete(file);
end

% constructs appropriate format for csv submission 
y_length = length(y_hat);
data_matrix = [(1 : y_length)', y_hat];
submission_title = {'ID', 'Prediction'};
submit = [submission_title; num2cell(data_matrix)];

% write matrix to csv file
cell2csv(file,submit);
end

