function [ x ] = feature_extract6( path_name, parameters)
%FEATURE_EXTRACT extracts percentages of ventricular, white brain mass and
%grey brain mass from an image. Limit1 and limit2 are parameters for
%gwv_weights that calculates the fractions. 

% calculate fractions
im = nii_read_volume(path_name);
temp = double(im < parameters.threshold*max(im(:)));
map = im > 0;
map = map.*temp;

% initialize x,y coordinates
x_init = floor(size(im,1)/2); 
y_init = floor(size(im,2)/2);

% define parameters 
z_max = parameters.z_max;
z_min = parameters.z_min;
x_min = parameters.lower_limit_x;
y_min = parameters.lower_limit_y;
x_max = parameters.upper_limit_x;
y_max = parameters.upper_limit_y;
ex = parameters.exponent;

% at every slice take a larger and larger rectangle of temp
x = 0;
parfor z = z_min:z_max
   x_l_l =  floor(x_init + (z / z_max)^ex * (x_min - x_init));
   x_u_l =  floor(x_init + (z / z_max)^ex * (x_max - x_init));
   y_l_l =  floor(y_init + (z / z_max)^ex * (y_min - y_init));
   y_u_l =  floor(y_init + (z / z_max)^ex * (y_max - y_init));
   
   s = map(x_l_l:x_u_l, y_l_l:y_u_l, z);
   x = x + sum(s(:));
end

x = [x, x^2, x^3];