function [ x ] = MLP3_feature_extract3_ar( path_name, parameters )
%FEATURE_EXTRACT Summary of this function goes here
%   Detailed explanation goes here

% for optimizing reasons (optimize brain after brain, not step after step -
% no need of loading another brain image each for-loop-iteration
load(path_name);
imo = im;


% Reduce image size
im = redImSize(imo,parameters.redIm);

% Filter image
% if parameters.filterOn
% %     sigma = 1;
% %     im = imgaussfilt3(im,sigma);
%     boxSize = [7 7 7];
%     im = medfilt3(im,boxSize);
% end

[x,y,z] = size(im);

% parameters
x_segments = parameters.x_segments; % TODO: optimize parameters
y_segments = parameters.y_segments;
z_segments = parameters.z_segments;

x_regions = floor(x/x_segments *(0:x_segments));
y_regions = floor(y/y_segments *(0:y_segments));
z_regions = floor(z/z_segments *(0:z_segments));

bins = parameters.bins; % TODO: optimize parameter

% Matrix indices start at 1 not 0
x_regions(1) = x_regions(1)+1;
y_regions(1) = y_regions(1)+1;
z_regions(1) = z_regions(1)+1;

% feature vectorML
x = zeros(x_segments*y_segments*z_segments, bins);

% Initzialize count variable
% TODO: optimize speed
count = 1;
for x_i = 1:x_segments
    for y_i=1:y_segments
        for z_i=1:z_segments
            % cut out chunk from image
            chunk = im(x_regions(x_i):x_regions(x_i + 1),...
                       y_regions(y_i):y_regions(y_i + 1),...
                       z_regions(z_i):z_regions(z_i + 1));
            % Adjust image intensity
                [cx,cy,cz] = size(chunk);
                chunkAdj = reshape(chunk,[cx,cy*cz]);
                mask = chunkAdj == 0;
            	chunkAdj(mask) = [];
                if ~isempty(chunkAdj)
                    chunkAdj = imadjust(chunkAdj);
                    chunk1d = chunkAdj(:);
                end
            h = histcounts(squeeze(chunk1d),bins);
            
            x(count, :) = h;
            
            count = count + 1;
        end
    end
end

x = x';
x = x(:)';
end