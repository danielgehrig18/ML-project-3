function [ x ] = feature_extract5( path_name , parameters)
%FEATURE_EXTRACT extracts percentages of ventricular, white brain mass and
%grey brain mass from an image. Limit1 and limit2 are parameters for
%gwv_weights that calculates the fractions. 

% calculate fractions
x = feature_extract3(path_name, parameters.segments);

lower_region = parameters.lower_region_l_limit:lower_region_u_limit;
upper_region = parameters.upper_region_l_limit:upper_region_u_limit;

first_mask = zeros(parameters(1),1);
first_mask(lower_region) = 1;

second_mask = zeros(parameters(1),1);
second_mask(upper_region) = 1;

x = [x*first_mask, x*second_mask];
