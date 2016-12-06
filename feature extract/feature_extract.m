function [ x ] = feature_extract( path_name , parameters)
%FEATURE_EXTRACT extracts percentages of ventricular, white brain mass and
%grey brain mass from an image. parameters(1) and parameters(2) are parameters for
%gwv_weights that calculates the fractions. 

% calculate fractions
im = nii_read_volume(path_name);

% im = preprocessing(im);

x = gwv_weights(im, parameters.lower_limit, parameters.upper_limit);
end

