X = zeros(1,138);       %Greymatter vector
Y = zeros(1,138);
threshold = 730;    % 0 to 1524 
limits = 5;

parfor i=1:138
    path_name = ['data/set_test/test_'  num2str(i) '.nii'];
    im = nii_read_volume(path_name);
%     imx = im;
    sz = size(im);
    

    
    %find mean intensities of images and a total mean
        for z=1:sz(3)   %area between top of brain and ventricles
            meanintensity = 0;
            for y=1:sz(2)
                pixels = 0;
                intensity = 0;
                for x=1:sz(1)     
                    if (im(x,y,z) > (threshold-limits) && im(x,y,z) < (threshold+limits))
                         X(1,i) = X(1,i) + 1;
%                         imx(x,y,z) = 0;
                    end
                end
            end
%         imshow([squeeze(imresize(im(:,:,z),3)),squeeze(imresize(imx(:,:,z),3))],[]);  
%         if (z == 125)
%             f = input('Press any key to continue');
%         end
        end
        
        Y(1,i) = -0.00428 *X(1,i) + 124.4;
end

y_length = length(Y');
data_matrix = ([(1:y_length)',Y']);
file = 'ML/project1/data/submit.csv';
% check if there is already a file with name 'submit.csv', if so delete it
if exist(file, 'file') == 2
    delete(file);
end

% write matrix to csv file
dlmwrite(file,data_matrix,'delimiter',',','-append');
