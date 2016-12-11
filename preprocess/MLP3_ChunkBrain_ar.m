function [ x ] = MLP3_ChunkBrain_ar( path_name, parameters,filterOn)
%FEATURE_EXTRACT Summary of this function goes here
%   Detailed explanation goes here

% Split brain into chunks, give the chunks back in a cell array

% for optimizing reasons (optimize brain after brain, not step after step -
% no need of loading another brain image each for loop iteration
im = nii_read_volume(path_name);

im = MLP3_redImSize( im,0.2,0.2,0.2 );

[x,y,z] = size(im);

% Filter image
if filterOn
%     sigma = 1;
%     im = imgaussfilt3(im,sigma);
    boxSize = [7 7 7];
    im = medfilt3(im,boxSize);
end

% parameters
x_segments = parameters.x_segments;
y_segments = parameters.y_segments;
z_segments = parameters.z_segments;

x_regions = floor(x/x_segments *(0:x_segments));
y_regions = floor(y/y_segments *(0:y_segments));
z_regions = floor(z/z_segments *(0:z_segments));

% bins = parameters.bins;

% Matrix indices start at 1 not 0
x_regions(1) = x_regions(1)+1;
y_regions(1) = y_regions(1)+1;
z_regions(1) = z_regions(1)+1;


% feature vector
% x = zeros(x_segments*y_segments*z_segments, bins);
% Save chunks in cell array (only uniform chunk sizes)
x = cell(x_segments^3,1);

% Initzialize count variable
count = 0;
for x_i = 1:x_segments
    for y_i=1:y_segments
        for z_i=1:z_segments
            % cut out chunk from image
            chunk = im(x_regions(x_i):x_regions(x_i + 1),...
                       y_regions(y_i):y_regions(y_i + 1),...
                       z_regions(z_i):z_regions(z_i + 1));
             count = count+1;
             x{count,1} = chunk;      
            
        end
    end
end

end

