function [ im_out ] = preprocessing( im_in )
%PREPROCESSING preprocesses by scaling the image slices so that each slice
%has the same maximum
im_in = double(im_in);

[x,y,z] = size(im_in);
scales = max(max(im_in,[],1),[],2);
scale_tensor = repmat(scales, x,y);
max_tensor = ones(x,y,z)*4360; % maximum intensity in all images

im_out = (scale_tensor./max_tensor).*im_in;

end

