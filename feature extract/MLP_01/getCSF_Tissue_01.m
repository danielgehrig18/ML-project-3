function [ CC ] = getCSF_Tissue_01( path_name , grey_limit, white_limit)
%FEATURE_EXTRACT extracts percentages of ventricular, white brain mass and
%grey brain mass from an image. Limit1 and limit2 are parameters for
%gwv_weights that calculates the fractions. 

% calculate fractions
raw = nii_read_volume(path_name);

% Get certain slice of the brain
% raw = im(:,:,115:116);

% all mass that has intensity above white_limit is white
% white = double(raw > white_limit);

% all mass between the limits grey_limit and white_limit is grey mass
grey_1 = raw > grey_limit;
grey_2 = raw < white_limit;
grey = grey_1.*grey_2;

% all mass above 0 and below grey_limit is ventricular ( = 0 is
% extracranial area)
% ventricle_1 = raw < grey_limit;
% ventricle_2 = raw > 0;
% ventricle = ventricle_1.*ventricle_2;

% Detect cerebrospinal fluid
CC = bwconncomp(grey);



% calculate fractions
% weights = [sum(white(:)),sum(grey(:)),sum(ventricle(:))];
% x = weights/sum(weights(:));

end

