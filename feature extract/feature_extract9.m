function [ x ] = feature_extract9( path_name, parameters)
%FEATURE_EXTRACT extracts percentages of ventricular, white brain mass and
%grey brain mass from an image. Limit1 and limit2 are parameters for
%gwv_weights that calculates the fractions. 

% calculate fractions
im = nii_read_volume(path_name);
sobel_x = fspecial('sobel');
sobel_y = sobel_x';

summer = ones(10);

proj = mean(im, 3);

ix = conv2(proj, sobel_x);
iy = conv2(proj, sobel_y);

ixx = conv2(ix.^2, summer);
iyy = conv2(iy.^2, summer);
ixy = conv2(ix.*iy,summer);

scores = ixx.*iyy-ixy.^2 - 0.08*(ixx+iyy).^2;

g = fspecial('gauss', )

x = sum(scores(:));

