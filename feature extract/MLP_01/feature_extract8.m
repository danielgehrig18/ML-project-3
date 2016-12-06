function [ x ] = feature_extract8( im , limit1, limit2)
%FEATURE_EXTRACT extracts percentages of ventricular, white brain mass and
%grey brain mass from an image. Limit1 and limit2 are parameters for
%gwv_weights that calculates the fractions. 

% calculate fractions
im = im(:,:,115:end);
x = get_tissue_fluid(im, limit1);
end

