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
[m,~] = size(X);
y_hat = zeros(m,3);
for k = 1:3
    y_hat(:,k) = predict(model{k}, X); 
end


%% check if there is already a file with name 'submit.csv', if so delete it
if exist(file, 'file') == 2
    delete(file);
end

% constructs appropriate format for csv submission 
y_length = length(y_hat);
y_ID = 0:3*y_length-1;
y_samples = repmat(0:y_length-1,3,1);
y_samples = reshape(y_samples,3*y_length,1);
y_hat = y_hat.';
y_hat = reshape(y_hat,3*y_length,1);
data_matrix = cell(3*y_length,4);
count = 0;
for k = 1:3*y_length
    count = count+1;
    data_matrix{k,1} = y_ID(k);
    data_matrix{k,2} = y_samples(k);
    if count == 1
        data_matrix{k,3} = 'gender';
    elseif count == 2
        data_matrix{k,3} = 'age';
    elseif count == 3
        data_matrix{k,3} = 'health';
        count = 0;
    end
    if y_hat(k)
        data_matrix{k,4} = 'true';
    else
        data_matrix{k,4} = 'false';
    end
end
% data_matrix = [(0 : 3*y_length-1)', y_samples, y_hat];
submission_title = {'ID','Sample','Label','Predicted'};
% submit = [submission_title; num2cell(data_matrix)];
submit = [submission_title; data_matrix];

% write matrix to csv file
cell2csv(file,submit);
end

