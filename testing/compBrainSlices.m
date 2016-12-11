% Compare several slices of sick and healthy brains

addpath('../ReadData3D_version1k/nii')

% V_1 = nii_read_volume('../data/set_train/train_103.nii');   % sick
V_1 = nii_read_volume('../data/set_train/train_1.nii');  % sick
V_2 = nii_read_volume('../data/set_train/train_2.nii');   % not sick
% V_2 = nii_read_volume('../data/set_train/train_153.nii');   % not sick
% V_2 = nii_read_volume('../data/set_train/train_208.nii');  % not sick
%%

% Reduce image size
xperc = 0.2;
yperc = 0.2;
zperc = 0.2;
V_1r = redImSize(V_1,[xperc,yperc,zperc]);
V_2r = redImSize(V_2,[xperc,yperc,zperc]);

% Filter image
sigma = 1;
% V_1r = imgaussfilt3(V_1r,sigma);
% V_2r = imgaussfilt3(V_2r,sigma);

% boxSize = [11 11 11];
% boxSize = [7 7 7];
% V_1r = medfilt3(V_1r,boxSize);
% V_2r = medfilt3(V_2r,boxSize);

% Choose axis to move along
[x,y,z] = size(V_1r);
slices = 16;
NoS = floor(z/slices);
axdir = 'z';
thresh = 0.2;

close all

for i = 6       %1:NoS
    % Select 16 slices
    Im1 = slice16(V_1r,i,axdir,thresh);
    Im2 = slice16(V_2r,i,axdir,thresh);
    
    % Adjust brightness
    Im1 = imadjust(Im1);
    minVal = min(Im1(:));
    imZero = Im1 == minVal;
    Im1(imZero) = 0;
    
    % Make unit8 greyscale
%     Im1 = uint8(Im1)*255;
%     Im2 = uint8(Im2)*255;
    
    figure(1)
    imshow(Im1,[])
    title('Sick Brain')
    
    figure(2)
    imshow(Im2,[])
    title('Healthy Brain')
    
    % Wait for button press
%     pause(5)
    
end