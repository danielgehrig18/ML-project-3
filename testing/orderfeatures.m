function [ orderedX ] = orderfeatures( X )
%orderfeatures is a function which takes generated features through
%feature_extract function and orders them according to a accending age in target.csv
%The return value is a matrix sorted matrix of pairs of age in the first
%column and feature value in the second column
Y = csvread('data/targets.csv');
orderedX = sortrows([Y , X],1);
orderedX(:,1) = [];
end

