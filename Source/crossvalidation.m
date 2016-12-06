function [ testvals ] = crossvalidation(CMP,Xtrain,ytrain,Wtrain,Xtest,ytest,Wtest)
%CROSSVALIDATION Summary of this function goes here
%   Detailed explanation goes here

global model1 model2 model3;

% template tree parameters
tree1 = model1.ModelParameters.LearnerTemplates{1};
tree2 = model2.ModelParameters.LearnerTemplates{1};
tree3 = model3.ModelParameters.LearnerTemplates{1};

% get parameters 
n_learn1 = model1.ModelParameters.NLearn;
n_learn2 = model2.ModelParameters.NLearn;
n_learn3 = model3.ModelParameters.NLearn;

method1 = model1.ModelParameters.Method;
method2 = model2.ModelParameters.Method;
method3 = model3.ModelParameters.Method;

learning_rate1 = model1.LearnRate;
learning_rate2 = model2.LearnRate;
learning_rate3 = model3.LearnRate;

if ~strcmp('Bag', method1)
    train_model1 = fitcensemble(Xtrain, ytrain(:,1), 'Method', method1, 'NumLearningCycles', n_learn1, 'Learners', tree1); 
else
    train_model1 = fitcensemble(Xtrain, ytrain(:,1), 'Method', method1, 'NumLearningCycles', n_learn1, 'LearnRate', learning_rate1, 'Learners', tree1);
end

if ~strcmp('Bag', method2)
    train_model2 = fitcensemble(Xtrain, ytrain(:,2), 'Method', method2, 'NumLearningCycles', n_learn2, 'Learners', tree2); 
else
    train_model2 = fitcensemble(Xtrain, ytrain(:,2), 'Method', method2, 'NumLearningCycles', n_learn2, 'LearnRate', learning_rate2, 'Learners', tree2);
end

if ~strcmp('Bag', method3)
    train_model3 = fitcensemble(Xtrain, ytrain(:,3), 'Method', method3, 'NumLearningCycles', n_learn3, 'Learners', tree3); 
else
    train_model3 = fitcensemble(Xtrain, ytrain(:,3), 'Method', method3, 'NumLearningCycles', n_learn3, 'LearnRate', learning_rate3, 'Learners', tree3);
end
    
yfit1 = predict(train_model1, Xtest);
yfit2 = predict(train_model2, Xtest);
yfit3 = predict(train_model3, Xtest);

testvals = 1/3 * trace(pdist2([yfit1, yfit2, yfit3], ytest, 'hamming'));
end