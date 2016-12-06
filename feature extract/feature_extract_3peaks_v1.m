function [ weights ] = feature_extract_3peaks_v1( path_name , parameters)
% Find number of grey values around specified center of gravity inside range
% (3 regions), using the grey intensity histogram

% Create grey value histogram (second input argument is not used, only a
% placeholder)
GH = feature_extract10(path_name,5000);

% Extract parameters
cg1 = parameters.cgone;
cg2 = parameters.cgtwo;
cg3 = parameters.cgthree;
ra1 = parameters.rangeone;
ra2 = parameters.rangetwo;
ra3 = parameters.rangethree;

% First peak
if (cg1-ra1) < 1
    idx1 = 1:(cg1+ra1);
else
    idx1 = (cg1-ra1):(cg1+ra1);
end
% Second peak
idx2 = (cg2-ra2):(cg2+ra2);
% Third peak
idx3 = (cg3-ra3):(cg3+ra3);

% Extract features from grey value histogram
ventricle = GH(idx1);
grey = GH(idx2);
white = GH(idx3);

% Count number of voxels in specified range
weights = [sum(ventricle(:)),sum(grey(:)),sum(white(:))];

% Normalize values
% weights = weights/sum(GH(:));
end

