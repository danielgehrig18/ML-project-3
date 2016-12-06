function [ numbers ] = random_sample( amount, limit )
%RANDOM_SAMPLE Summary of this function goes here
%   Detailed explanation goes here
numbers = randi([1 limit],1,amount);

end

