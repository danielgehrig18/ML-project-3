% Generate a grey intensity histogram of all brains -> faster optimization
% of feature extracts that only need grey intensity as input information
% (loading and evaluating a .nii image takes longer than doing the same
% for a grey intensity histogram)
clear
clc

% Get nii files
% folder = '../data/set_train/';
folder = '../data/set_test/';
extension = '*.nii';
listing = dir(strcat(folder,extension));

% Number of brains
NoB = length(listing);

% Get folder order in listing array
folder_order = zeros(NoB,1);

for i = 1:NoB
    file_name = listing(i).name;
    % skip system files (.etc) (only needed if not looking for .nii files
    % only)
    if file_name(1) == '.'
        continue
    end
    % find file number (file of the form 'test_3.nii'
    parse1 = strsplit(file_name, '.');
    parse2 = strsplit(parse1{1}, '_');
    file_number = str2num(parse2{2});
    
    folder_order(i) = file_number;
end

% Create grey histogram safe cell array
A = cell(NoB,1);

% Generate grey histograms of all brains in set_train
disp('START GREY HISTOGRAM GENERATION')
tic

parfor i = 1:NoB
    path_name = strcat(folder,listing(i).name);
    [greyHisto,~] = greyhisto(path_name);
    A{i} = greyHisto;
    disp(['Generated grey histogram of brain number' num2str(folder_order(i))])
end

% Get elapsed time
elapsedTime = toc;
% Convert to hours (from seconds, default)
elapsedTimeh = elapsedTime/3600;

disp(['The generation took ' num2str(elapsedTimeh) ' hours or ' num2str(elapsedTime) ...
    ' seconds.']);

% Save histograms
% Change path to greyhisto_train folder (in data)
disp('Saving histograms ...')

% cd('../data/greyhisto_train')
cd('../data/greyhisto_test')
for i = 1:NoB
    file_name = listing(i).name;
    parse1 = strsplit(file_name,'.');
    file_name = strcat(char(parse1{1}),'.mat');
    temp_GH = A{i};
    save(file_name,'temp_GH')
end
% Go back to preprocess folder
cd('../../preprocess')

disp('Histograms saved successfully!')
