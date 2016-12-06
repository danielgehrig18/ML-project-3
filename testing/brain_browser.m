% show brains
    hold on;
    threshold = 1000;
    final_image = zeros(352,416,3);
    V_1 = nii_read_volume('data/set_train/train_57.nii');   %Age: 18
    V_2 = nii_read_volume('data/set_train/train_227.nii');  %Age: 45
    V_3 = nii_read_volume('data/set_train/train_65.nii');   %Age: 70
    V_4 = nii_read_volume('data/set_train/train_160.nii');  %Age: 90
    i = 105;
    while 1
        str = input('up = 1 // down = 0    : ');
        switch str
            case 1
                i = i+1
            case 0
                i = i-1
            case 2
                i = i+5
            case 9
                i = i-5
            case 3
                i = i+10
            case 8
                i = i-10
            otherwise
                threshold = (str);
        end
        im = double([V_1(i,:,:), V_2(i,:,:);
                 V_3(i,:,:), V_4(i,:,:)]);
             
        im = uint8(im/max(im(:))*256);
        
        [BW] = edge(im,'Sobel', threshold);
        
        im(BW==1) = 0;
        
        final_image(:,:,1) = im;
        final_image(:,:,2) = im;
        
        im(BW==1) = 255;
        
        final_image(:,:,3) = im;
             
        im = repmat(im,1,1,3);
        
        im(BW==1) = 0;
             
        imshow(uint8(final_image));
      
end

    