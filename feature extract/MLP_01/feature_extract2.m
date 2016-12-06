function [ x ] = feature_extract2( path_name , parameters)
%FEATURE_EXTRACT Summary of this function goes here
%   Detailed explanation goes here
% Divide grey histogram in three parts
if ischar(path_name)
    % path_name is the path to the .nii brain image
    im = nii_read_volume(path_name); 
    x = gwv_weights_modar(im, parameters.lower_limit, parameters.upper_limit);
elseif ismatrix(path_name)
    % path_name is the grey intensity histogram of the brain
    im = path_name(:);
    x = gwv_weights_greyh(im, parameters.lower_limit, parameters.upper_limit);
end
            
% Old version: Why only to end/2 ??
% x = gwv_weights(im(:,:,end/2), ...      
%                 parameters.lower_limit, ...
%                 parameters.upper_limit);
end