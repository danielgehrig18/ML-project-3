function [ weights ] = getCSF_Tissue_03( GH , parameters)
% Find number of grey values around specified center of gravity inside range
% (3 regions), using the grey intensity histogram

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

ventricle = GH(idx1);
grey = GH(idx2);
white = GH(idx3);

% calculate fractions
weights = [sum(ventricle(:)),sum(grey(:)),sum(white(:))];
% weights = weights/sum(GH(:));
end

