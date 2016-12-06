function [ histo, greyVals ] = greyhisto( path_name )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

if ischar(path_name)
    addpath('../ReadData3D_version1k/nii');

    % Load image
    im = nii_read_volume(path_name);
else
    im = path_name;
end

% Find maximum and minimum grey value
maxGreyVal = max(max(max(im,[],3),[],2),[],1);
minGreyVal = min(min(min(im,[],3),[],2),[],1);
% Omit extracranial matter
if minGreyVal == 0
    minGreyVal = 1;
end

% Create grey value vector, omit 0 values as no image information
greyVals = minGreyVal:1:maxGreyVal;
greyVals = greyVals(:);

% Get number of each grey value
histo = zeros(maxGreyVal,1);
for i = minGreyVal:1:maxGreyVal
    idx = (im == i);
    histo(i) = sum(idx(:));
%     idx = find(im == i);
%     histo(i) = length(idx);
end
end

