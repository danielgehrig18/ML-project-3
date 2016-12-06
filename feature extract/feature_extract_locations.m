function [ x_feature ] = feature_extract_locations( path_name, parameters )
%FEATURE_EXTRACT_CEREBRAL_HEMISPHERE Summary of this function goes here
%   Detailed explanation goes here
x_distances = [];
    im = double(nii_read_volume(path_name));
    imx = (im < (parameters.target + parameters.offset) & im > (parameters.target - parameters.offset));
    
    k = find(imx);
    [x,y,z] = ind2sub(size(imx),k);
    
    x_distances = [(x-86).^2,(y-104).^2,(z-86).^2];

x_feature = mean(x_distances);