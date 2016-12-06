function [ weights ] = gwv_weights_greyh( raw, grey_limit, white_limit )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
    % Get number of voxels between certain limits (originally: white, grey
    % matter and ventricles)
    % raw is a column vector containing the number of voxels at different
    % grey values
% Set top limit (above: only noise)
toplimit = 1750;

white = raw(white_limit+1:toplimit);
grey = raw(grey_limit+1:white_limit);
ventricle = raw(1:grey_limit);

weights = [sum(white(:)),sum(grey(:)),sum(ventricle(:))];
% Normalisation
weights = weights/sum(weights(:));

end

