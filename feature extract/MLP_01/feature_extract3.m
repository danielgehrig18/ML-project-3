function [ x ] = feature_extract3( path_name , parameters)
% Divide grey histogram in several parts of the same length and count
% number of voxels in each part

x = zeros(1,parameters.segments);

% calculate fractions
if ischar(path_name)
    im = double(nii_read_volume(path_name));
    im = im*parameters.segments/max(im(:));
    
    for i=0 : parameters.segments - 1
    p_region = im < i + 1;
    m_region = im > i;
    
    total = p_region .* m_region;
    
    x(i+1) = sum(total(:));
    end
elseif ismatrix(path_name)
    % If a grey inensity histogram is handed over
    im = path_name(:);
    parts = length(im)/parameters.segments;
    for i = 0:parameters.segments-1
        llim = floor(i*parts)+1;
        ulim = floor((i+1)*parts);
        part_vals = im(llim:ulim,1);
        x(i+1) = sum(part_vals);
    end   
else
    error('Wrong variable type')
end
end

