function [ w ] = feature_extract_3peaks( path_name , parameters)
% Find number of grey values around specified center of gravity inside range
% (3 regions), using the grey intensity histogram

% Create grey value histogram (second input argument is not used, only a
% placeholder)

% best parameters through optimization:
%       cgone: 220
%       cgtwo: 1450
%       cgthree: 750
%       rangeone: 35
%       rangetwo: 50
%       rangethree: 110

GH = feature_extract10(path_name,parameters);
GH = GH/sum(GH(:));

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
w = [sum(ventricle(:)),sum(grey(:)),sum(white(:))];
w = [w, w(2)^3, w(1)*w(3), w(2)*w(3), w(2)^4, w(2)^3*w(3)];
end

