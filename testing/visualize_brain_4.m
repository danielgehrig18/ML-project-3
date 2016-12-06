addpath('feature extract', 'preprocess','ReadData3D_version1k/nii');
h = 8;
% show brains
hold on;
    threshold = 1000;
    lower_threshold = 800;
    
    final_image = zeros(176,208);%528,624,3);
    
    numbers = random_sample(9, 278);
    
    V_1 = nii_read_volume(strcat('data/set_train/train_',num2str(numbers(1)),'.nii'));   %Age: 18
    %V_2 = nii_read_volume(strcat('data/set_train/train_',num2str(numbers(2)),'.nii'));   %Age: 18
    %V_3 = nii_read_volume(strcat('data/set_train/train_',num2str(numbers(3)),'.nii'));   %Age: 18
    %V_4 = nii_read_volume(strcat('data/set_train/train_',num2str(numbers(4)),'.nii'));   %Age: 18
    %V_5 = nii_read_volume(strcat('data/set_train/train_',num2str(numbers(5)),'.nii'));   %Age: 18
    %V_6 = nii_read_volume(strcat('data/set_train/train_',num2str(numbers(6)),'.nii'));   %Age: 18
    %V_7 = nii_read_volume(strcat('data/set_train/train_',num2str(numbers(7)),'.nii'));   %Age: 18
    %V_8 = nii_read_volume(strcat('data/set_train/train_',num2str(numbers(8)),'.nii'));   %Age: 18
    %V_9 = nii_read_volume(strcat('data/set_train/train_',num2str(numbers(9)),'.nii'));   %Age: 18
    %V_1 = imgaussfilt3(V_1,3);
    %V_1 = V_1(43:125, 47:170,:);
    
    CC = feature_extract7(V_1, 400, 3,8);
    i = 100;
    
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
        a = V_1;
        b = a;
        numPixels = cellfun(@numel,CC.PixelIdxList);
        [biggest,idx] = max(numPixels);
        a(CC.PixelIdxList{idx}) = 2000;
        
        im = double([b(:,:,i)]);%, V_2(:,:,i), V_3(:,:,i);
            %V_6(:,:,i), V_5(:,:,i), V_4(:,:,i);
            %V_7(:,:,i), V_8(:,:,i), V_9(:,:,i)]);
        im = uint8(im/max(im(:))*256);
        im = repmat(im, 1,1,3);
        
        im(:,:,1) = a(:,:,i);
        
        
        
        % show image
        imshow(uint8(im));
        
    end

    