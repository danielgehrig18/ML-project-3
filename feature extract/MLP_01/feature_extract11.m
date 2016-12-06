function [ x ] = feature_extract11( path_name, parameters )
%FEATURE_EXTRACT11 Summary of this function goes here
%   Detailed explanation goes here

histo = feature_extract10(path_name, parameters);
segm = histo(500:1200);

[~, max_id] = max(segm);

up = max_id + parameters.up;
down = max_id - parameters.down;

area = sum(segm(down:up));

x = [area, max_id];

end

