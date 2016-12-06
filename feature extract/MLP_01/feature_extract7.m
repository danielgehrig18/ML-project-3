function [ x ] = feature_extract7( path_name, parameters)
%FEATURE_EXTRACT7 Summary of this function goes here
%   Detailed explanation goes here

im = nii_read_volume(path_name);
threshold = parameters.threshold * max(im(:));

im = imgaussfilt3(im, parameters.filter_variance);

z = im > 0;

search_space = im < threshold;
search_space = z.*search_space;

blobs = bwconncomp(search_space, parameters.connectivity);
numPixels = cellfun(@numel,blobs.PixelIdxList);

x = sort(numPixels);
x = x(end);
end