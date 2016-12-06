function [ x ] = getCSF_Tissue_02( path_name , limit_vec)
%FEATURE_EXTRACT extracts percentages of ventricular, white brain mass and
%grey brain mass from an image. Limit1 and limit2 are parameters for
%gwv_weights that calculates the fractions. 

% calculate fractions
im = nii_read_volume(path_name);

% select most upper part, above ventricles
raw = im(:,:,115:end);

% Get limits
if length(limit_vec) == 6
    bot_black = limit_vec(1);
    top_black = limit_vec(2);
    bot_grey = limit_vec(3);
    top_grey = limit_vec(4);
    bot_white = limit_vec(5);
    top_white = limit_vec(6);
else
    error('Wrong number of limits')
end

% all mass that has intensity above white_limit is white
white_1 = raw > bot_white;
white_2 = raw < top_white;
white = double(white_1.*white_2);

% all mass between the limits grey_limit and white_limit is grey mass
grey_1 = raw > bot_grey;
grey_2 = raw < top_grey;
grey = double(grey_1.*grey_2);

% all mass above 0 and below grey_limit is ventricular ( = 0 is
% extracranial area)
ventricle_1 = raw < top_black;
ventricle_2 = raw > bot_black;
ventricle = double(ventricle_1.*ventricle_2);

% calculate fractions
weights = [sum(white(:)),sum(grey(:)),sum(ventricle(:))];
x = weights/sum(weights(:));
end

