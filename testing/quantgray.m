function[rgbim, countlabel] = quantgray(im, levels)

% Compute thresholds according to Otsu's method (Number of thresholds =
% levels)
thresh = multithresh(im,levels);
% Segment the image according to the thresholds
labelim = imquantize(im,thresh);
% Count the voxels in each segment
countlabel = zeros(levels,1);
for k = 1:levels
    tempcl = labelim == k;
    countlabel(k,1) = sum(tempcl(:));
end
% Create rgb image (each segment is colored differently)
rgbim = label2rgb(labelim);

end

