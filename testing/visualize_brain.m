% show brains
% sick brains are:
% 9, 14, 16, 17, 18, 28, 31, 37, 38, 49, 52, 55, 56, 59, 61, 66, 72, 78, 
% 79, 97, 103, 109, 116, 118, 122, 126, 131, 133, 134, 136, 139, 144, 
% 152, 157, 158, 159, 162, 163, 168, 176, 178, 179, 181, 182, 188, 189,
% 191, 195, 197, 198, 203, 206, 207, 219, 221, 226, 230, 242, 243, 250, 
% 256, 257, 258, 261, 262, 266, 267 

close all

V_1 = nii_read_volume('data/set_train/train_103.nii');   % sick
V_2 = nii_read_volume('data/set_train/train_158.nii');  % sick
V_3 = nii_read_volume('data/set_train/train_153.nii');   % not sick
V_4 = nii_read_volume('data/set_train/train_208.nii');  % not sick
i = 1;

block = [V_1];

while true
    slice = V_1(:,:,i);
    [th, m] = multithresh(slice, 2);
    m
    l = imquantize(slice, th);
    
    imshow(label2rgb(l));
    
    %imshow(imrotate(block(:,:,i), 270),[]);
    if waitforbuttonpress
        i = i-2
    else
        i = i+2
    end
end

    