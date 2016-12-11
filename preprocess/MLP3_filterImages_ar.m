function [ x ] = MLP3_filterImages_ar( path_name, parameters )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
% Load image
imo = nii_read_volume(path_name);

% Filter image
boxSize = parameters.boxSize;
im = medfilt3(imo,boxSize);

% Save filtered image
parse = strsplit(path_name,'/');
parse2 = strsplit(parse{end},'.');
file_name = strcat(parse2{1},'_filtered.mat');
save(file_name,'im')

% Give something back
x = 0;

end

