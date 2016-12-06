% Show what was detected in the different categories

addpath('../data/set_train','../ReadData3D_version1k/nii')
% current optimal values
white_limit = 600;
grey_limit = 150;

V_1 = nii_read_volume('data/set_train/train_3.nii');
raw = V_1;

% white = (raw > white_limit)*1000;
grey_1 = raw > grey_limit;
grey_2 = raw < white_limit;
grey = (grey_1.*grey_2);
% ventricle_1 = raw < grey_limit;
% ventricle_2 = raw > 0;
% ventricle = (ventricle_1.*ventricle_2)*1000;

% find connected areas
CC = bwconncomp(grey);

% convert cell array to label matrix
L = labelmatrix(CC);

h = figure(1);
for i = 1:10:176 % 115 <- above ventricle (176x208x176)
%     imshow([squeeze(V_1(i,:,:)),squeeze(grey(i,:,:))],[])
    imshow(label2rgb(squeeze(L(:,:,i))));
    truesize(h,[550 550])
    waitforbuttonpress
end
close
