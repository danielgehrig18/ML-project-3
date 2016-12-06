% Grey intensity histogram evaluation
clear


% ------------------------------------------------------
% Attention! 
% dir does not give back the found files in the same order as
% they are sorted in the set_train folder!
% -> Either rearrange the listing array or take it into account
% differently!
% ------------------------------------------------------

% Get nii files
folder = '../data/set_train/';
extension = '*.nii';
listing = dir(strcat(folder,extension));

% Create struct saving the grey inensity histograms and the grey value
% vectors
field1 = 'greyHisto';
field2 = 'greyVals';
value1 = {};
value2 = {};

greyHisto_struct = struct(field1,value1,field2,value2);

% Struct containing greyHisto_struct of all different MRI's
field1 = 'GH';
value1 = {};
all_struct = struct(field1,value1);

% Split brain in parts in direction of dimension
dimension = 3;
parts = 1;

% Numbers of first and last evaluated MRI
fmri = 1;
lmri = 4;
% MRI counter (index)
mri = 0;

for p = fmri:lmri %1:length(listing)
    mri = mri+1;
    path_name = strcat(folder,listing(p).name);
    split_brain = brain_splitter(path_name,dimension,parts);
    for i = 1:parts 
        im = split_brain{i};
        [greyHisto, greyVals] = greyhisto(im);
        greyHisto_struct(i).greyHisto = greyHisto;
        greyHisto_struct(i).greyVals = greyVals;
    end
    all_struct(mri).GH = greyHisto_struct;
end

% %%
% % plot Histogram
% % Find overall maximum grey value
% abs_max_old = 0;
% for i = 1:5
%     abs_max_new = max(greyHisto_struct(i).greyVals);
%     if abs_max_old < abs_max_new
%         abs_max_old = abs_max_new;
%     end
% end
% 
% abs_max = double(abs_max_old);
% %%
% 
% % Scale
% for i = 1:5
%     ind_max = double(max(greyHisto_struct(i).greyVals));
%     greyHisto_struct(i).greyVals = double(greyHisto_struct(i).greyVals)/ind_max*abs_max;
% end
% 
%%

close all
for i = 1:parts
figure(i)
for p = 1:mri
    h1 = plot(all_struct(p).GH(i).greyVals,all_struct(p).GH(i).greyHisto);
    hold on
end
title('Grey Intensity Graph')
% legend('Part 1','Part 2','Part 3','Part 4','Part 5','Part 6','Part 7','Part 8')
% legend('20','25','61','84','23')
legend('20','25','61','84')
xlabel('Grey values')
ylabel('Intensity')
xlim([0 1750]);
hold off
end



