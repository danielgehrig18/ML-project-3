function [ split_im ] = brain_splitter( path_name, dimension, parts )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

if ischar(path_name)
    addpath('../ReadData3D_version1k/nii');

    % Load image
    im = nii_read_volume(path_name);
else
    im = path_name;
end

split_size = size(im,dimension);

remainder = mod(split_size,parts);
divider = floor(split_size/parts);

if remainder == 0
    split_im = cell(parts,1);
else
    split_im = cell(parts,1);
end

for i = 1:parts
    switch dimension
        case 1
            split_im{i} = im((i-1)*divider+1:i*divider,:,:);
        case 2
            split_im{i} = im(:,(i-1)*divider+1:i*divider,:);
        case 3
            split_im{i} = im(:,:,(i-1)*divider+1:i*divider);
        otherwise
            error('No dimension of the image')
    end    
end

if remainder > 0
    switch dimension
        case 1
            split_im{end} = im(divider*parts:divider*parts+remainder,:,:);
        case 2
            split_im{end} = im(:,divider*parts:divider*parts+remainder,:);
        case 3
            split_im{end} = im(:,:,divider*parts:divider*parts+remainder);
        otherwise
            error('No dimension of the image')
    end    
end

    

end

