function [ feature ] = feature_extract1( path_name, parameters )
%FEATURE_EXTRACT Summary of this function goes here
%   Detailed explanation goes here

% for optimizing reasons (optimize brain after brain, not step after step -
% no need of loading another brain image each for loop iteration
if ischar(path_name)
    im = nii_read_volume(path_name);
elseif ismatrix(path_name)
    im = path_name;
else
    error('Wrong input data: path_name must be string or matrix')
end

[x,y,z] = size(im);

% parameters
x_segments = parameters.x_segments;
y_segments = parameters.y_segments;
z_segments = parameters.z_segments;

x_regions = floor(x/x_segments *(0:x_segments));
y_regions = floor(y/y_segments *(0:y_segments));
z_regions = floor(z/z_segments *(0:z_segments));

% Matrix indices start at 1 not 0
x_regions(1) = x_regions(1)+1;
y_regions(1) = y_regions(1)+1;
z_regions(1) = z_regions(1)+1;


% feature vector
feature_vec = zeros(1,x_segments*y_segments*z_segments);

% Initzialize count variable
count = 0;

for x_i = 1:x_segments
    for y_i=1:y_segments
        for z_i=1:z_segments
            count = count+1;
            % cut out chunk from image
            chunk = im(x_regions(x_i):x_regions(x_i + 1),...
                       y_regions(y_i):y_regions(y_i + 1),...
                       z_regions(z_i):z_regions(z_i + 1));
            
            % take mean of intensity in chunk       
            m = mean(chunk(:)); 
%             if m~=0
%                 % Save mean if not equal to 0
%                 feature_mat(x_i,y_i,z_i) = m;
%             end
            feature_vec(count) = m;
        end
    end
end

feature = feature_vec;
% Create feature array, only save non zero values
% -> not valid, as every feature vector must have the same number of
% elements for the regression!

% iszero = feature_mat == 0;
% feature = zeros(sum(iszero(:)),1);
% 
% idx = find(feature_mat ~= 0);
% n = 0;
% 
% for i = idx'
%     n = n+1;
%     feature(n) = feature_mat(i);
% end
end

