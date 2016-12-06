function model = train( x_folder, y_file, fun, parameters )
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

% load classification tree calculated through optimization.
t = load('classification_tree.mat');

% train bagging model. Trained through optimization
model = fitcensemble(X, y, 'Method', 'Bag', 'NumLearningCycles', 11, 'Learners', t.tree);