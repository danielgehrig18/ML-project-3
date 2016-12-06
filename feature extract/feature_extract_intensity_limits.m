%function [ x ] = feature_extract_intensity_limits( path_name, threshold )
%FEATURE_EXTRACT_CEREBRAL_HEMISPHERE Summary of this function goes here
%   Detailed explanation goes here
clear all

relativethreshold = 2.6;        %260% of the mean

X = zeros(1,278);
%threshold = 462;    % 0 to 1524 // optimal threshold 462 determined through numerical iteration

parfor i=1:278
    path_name = ['data/set_train/train_'  num2str(i) '.nii'];
    im = double(nii_read_volume(path_name));
    sz = size(im);
    meanintensities = zeros(1,sz(3)); %meanintensities for eacht considered slice
    
    %find mean intensities of images and a total mean
    slices = 0;

        for z=115:sz(3)   %area between top of brain and ventricles
            meanintensity = 0;
            for y=1:5:sz(2)
                pixels = 0;
                intensity = 0;
                for x=1:5:sz(1)     
                    if (im(x,y,z) ~= 0)
                        intensity = intensity + im(x,y,z);    %take sample intensity
                        pixels = pixels + 1;
                        meanintensity = intensity/pixels;                     
                    end
                end
            end 
                    slices = slices+1;
                    meanintensities(1,z) = meanintensity; %vector of mean intensities in % of scale
        end
totalmeanintensity = 0;
slices = 0;
for j=1:176
    if (meanintensities(1,j)~=0)
        totalmeanintensity = totalmeanintensity + meanintensities(1,j);
        slices = slices+1;
    end
end
    totalmeanintensity = totalmeanintensity / slices;
    
    %find cerebrespinal fluid volume
    cerebral_hemisphere_tissue_volume = 0;
    nonzeros = 0;
    for z=115:sz(3)   %area between top of brain and ventricles
        area = 0;
        threshold = meanintensities(1,z)*relativethreshold;
        for y=1:sz(2)
            for x=1:sz(1)     
                if (im(x,y,z) ~= 0)
                    nonzeros = nonzeros + 1;    %count entire brain volume
                end
                
                if (im(x,y,z) < threshold && im(x,y,z) > 0)
                   area = area + 1;         %found pixel above threshold --> count it to area
                   im(x,y,z) = 0;           %short visualization
                end
            end
        end
            cerebral_hemisphere_tissue_volume = cerebral_hemisphere_tissue_volume + area*1; %thickness of slice = 1
    end
    normalized_tissue = cerebral_hemisphere_tissue_volume/nonzeros;
    X(1,i) = normalized_tissue;
end

Y = csvread('data/targets.csv');
scatter(Y,X);
hold on;

y_dash = mean(Y);
x_dash = mean(X);

Sxx = 0;
Syy = 0;
Sxy = 0;

yCalc = zeros(1,278);
Error = 0;

for j=1:278
    Sxx = Sxx + (X(1,j) - x_dash)^2;
    Syy = Syy + (Y(j,1) - y_dash)^2;
    Sxy = Sxy + (X(1,j) - x_dash)*(Y(j,1) - y_dash);
end

m = Sxy/Sxx;
b = y_dash - m*x_dash;

for j=1:278
    yCalc(1,j) = m * X(1,j) + b;
    Error = Error + (yCalc(1,j) - Y(j,1))^2;
end

disp(['The MEAN-Error is: ' num2str(sqrt(Error/278)) ' years @ Threshold: ' num2str(relativethreshold*100) '%']);

plot(yCalc,X)