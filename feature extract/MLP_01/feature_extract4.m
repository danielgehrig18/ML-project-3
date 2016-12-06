function [ x ] = feature_extract4( path_name , parameters)
%FEATURE_EXTRACT extracts percentages of ventricular, white brain mass and
%grey brain mass from an image. Limit1 and limit2 are parameters for
%gwv_weights that calculates the fractions. 

% calculate fractions
im = nii_read_volume(path_name); 
x = gwv_weights(im, ...      
                parameters.lower_limit, ...
                parameters.upper_limit);
x = x(2:end);
end

