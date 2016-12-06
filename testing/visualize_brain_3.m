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
    
    i = 100;
    
    t_b_x = 69;
    t_b_y = 46;
    
    b_b_x = 69;
    b_b_y = 88;
    
    x_s_1 = 60;
    x_s_2 = 60;
    
    y_s_1 = 40;
    y_s_2 = 40;
    
    while 1
        str = input('up = 1 // down = 0    : ');
        switch str
            case 5
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
        
        im1 = double([V_1(:,:,i)]);%, V_2(:,:,i), V_3(:,:,i);
            %V_6(:,:,i), V_5(:,:,i), V_4(:,:,i);
            %V_7(:,:,i), V_8(:,:,i), V_9(:,:,i)]);
        im1 = uint8(im1/max(im1(:))*256);
        im = im1;    
        
            
        im = imguidedfilter(im);    
             
        
        
        BW = im < threshold;
        
        
                
        % construct color image
        im(BW==1) = 0;
        final_image(:,:,1) = im;
        final_image(:,:,2) = im;
        im(BW==1) = 255;
        final_image(:,:,3) = im;
        im(BW==1) = 0;
             
        final_image(t_b_y:t_b_y + y_s_1, t_b_x:t_b_x + x_s_1, 1) = 255;
        final_image(b_b_y:b_b_y + y_s_2, b_b_x:b_b_x + x_s_2, 2) = 255;
        
        % show image
        imshow(uint8([final_image, repmat(im1,1,1,3)]));
        
        % get new parameters
        box1 = BW(t_b_y:t_b_y + y_s_1, t_b_x:t_b_x + x_s_1);
        box2 = BW(b_b_y:b_b_y + y_s_2, b_b_x:b_b_x + x_s_2);

        [s_x_1, s_y_1, m_1] = abc(box1)        
        [s_x_2, s_y_2, m_2] = abc(box2)
        
        h_x_1 = 3;%*s_x_1/(s_x_1-1);
        h_y_1 = 3;%*s_y_1/(s_y_1-1);
        h_x_2 = 3;%*s_x_2/(s_x_2-1);
        h_y_2 = 3;%*s_y_2/(s_y_2-1);
        
        if ~isnan(m_1(1)) && ~isnan(m_1(2)) && ~isnan(s_x_2) && ~isnan(s_y_2)&& ...
                ~isnan(m_2(1)) && ~isnan(m_2(2)) && ~isnan(s_x_1) && ~isnan(s_y_1) 
            t_b_x = t_b_x+m_1(1)-h_x_1*s_x_1;
            t_b_y = t_b_y+m_1(2)-h_y_1*s_y_1;
            
            x_s_2 = floor(2*h_x_2*s_x_2);
            x_s_1 = floor(2*h_x_1*s_x_1);
            
            y_s_1 = floor(2*h_y_1*s_y_1);
            y_s_1 = min([t_b_y+y_s_1, 88-t_b_y]);
            
            b_b_x = b_b_x+floor(m_2(1)-h_x_2*s_x_2);
            
            
            b_b_y = b_b_y+floor(m_2(2)-h_y_2*s_y_2);
            y_s_2 = floor(2*h_y_2*s_y_2);
            
            y_max = b_b_y + y_s_2;
            
            b_b_y = max([88, b_b_y]);
            
            y_s_2 = y_max - b_b_y;
        end
        
    end

    