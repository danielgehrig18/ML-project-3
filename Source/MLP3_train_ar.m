function model = MLP3_train_ar( x_folder, y_file, fun, parameters )
%TRAIN_B Trains a classification model with Matlab function LinearModel.fit
%   Args:   x_folder:   folder with all the training data for X
%           y_file:     file with the training data for y
%           fun:        function to be used for the feature extraction
%           parameters: struct containing all relevant arguments to execute
%                       fun
%
%   Return: model: object of type LinearModel containing a trained model
%                  with training set
%           X:     Data matrix (# datapoints) x (# features) 

% loads targets
y = csvread(y_file);

% generates #datapoints x (#features) data matrix
X = generate_X(x_folder, fun, parameters); 

% train bagging model. Trained through optimization
kernel = 'rbf';
model = cell(3,1);
b = [80 50 50];
C = [800 400 6000];
for k = 1:3
    model{k} = fitcsvm(X,y(:,k),'Standardize',true,'KernelFunction',kernel,...
                'BoxConstraint',C(k),'KernelScale',b(k));
end
end