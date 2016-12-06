function [ x ] = feature_extract12( path_name, parameters )
%FEATURE_EXTRACT11 Summary of this function goes here
%   Detailed explanation goes here
%   parameters: target = line of first peaks
%
%   optimal: target ~ 335 --> RMS 13,3937
histo = feature_extract10(path_name, parameters);
x=0;
for tar=(parameters.target-20):(parameters.target+20)
    x = x + histo(tar);                    %to be optimized 
end
end

