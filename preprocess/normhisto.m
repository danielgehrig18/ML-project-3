function [ ngh ] = normhisto( gh )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
% Normalize grey values between 0 and 1
gh_len = length(gh);
ngh = 1:gh_len;
ngh = ngh/gh_len;
ngh = ngh';

end

