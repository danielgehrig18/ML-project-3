function [ testvals ] = crossvalidation(CMP,Xtrain,ytrain,Wtrain,Xtest,ytest,Wtest)
%CROSSVALIDATION Summary of this function goes here
%   Detailed explanation goes here

global model;

% template tree parameters
tree = model.ModelParameters.LearnerTemplates{1};

% get parameters 
n_learn = model.ModelParameters.NLearn;
method = model.ModelParameters.Method;
learning_rate = model.LearnRate;

if ~strcmp('Bag', method)
    train_model = fitcensemble(Xtrain, ytrain, 'Method', method, 'NumLearningCycles', n_learn, 'Learners', tree); 
else
    train_model = fitcensemble(Xtrain, ytrain, 'Method', method, 'NumLearningCycles', n_learn, 'LearnRate', learning_rate, 'Learners', tree);
end
    
yfit = predict(train_model, Xtest);

testvals = 1/3 * trace(pdist2(yfit, ytest, 'hamming'));
end