function [ weights ] = gwv_weights( raw, grey_limit, white_limit )
%COLOR_AREAS gets fractions of white and grey brain mass and ventricular
%brain mass

% all mass that has intensity above white_limit is white
white = double(raw > white_limit);

% all mass between the limits grey_limit and white_limit is grey mass
grey_1 = raw > grey_limit;
grey_2 = raw < white_limit;
grey = grey_1.*grey_2;

% all mass above 0 and below grey_limit is ventricular ( = 0 is
% extracranial area)
ventricle_1 = raw < grey_limit;
ventricle_2 = raw > 0;
ventricle = ventricle_1.*ventricle_2;

% calculate fractions
weights = [sum(white(:)),sum(grey(:)),sum(ventricle(:))];
weights = weights/sum(weights(:));
end

