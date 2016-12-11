function [ smallIm ] = redImSize( im,redPerc )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
% Reduce the image size, get rid of the black surrounding (tighten the box
% that surrounds the brain)

[x,y,z] = size(im);

xperc = redPerc(1);
yperc = redPerc(2);
zperc = redPerc(3);
% Number of voxels to omit on each side
xpix = floor(xperc*x/2);
ypix = floor(yperc*y/2);
zpix = floor(zperc*z/2);

smallIm = im(xpix+1:end-xpix,ypix+1:end-ypix,zpix+1:end-zpix);

end

