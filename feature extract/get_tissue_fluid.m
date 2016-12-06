function [ weights ] = get_tissue_fluid( raw, grey_limit)
%COLOR_AREAS gets fractions of white and grey brain mass and ventricular
%brain mass

% all mass between the limits grey_limit and white_limit is grey mass
grey = raw > grey_limit;

% all mass above 0 and below grey_limit is ventricular ( = 0 is
% extracranial area)
ventricle_1 = raw < grey_limit;
ventricle_2 = raw > 0;
ventricle = ventricle_1.*ventricle_2;

% calculate fractions
weights = [sum(grey(:)),sum(ventricle(:))];
weights = weights/sum(weights(:));
end

