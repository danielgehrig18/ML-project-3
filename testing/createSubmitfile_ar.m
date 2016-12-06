% Create csv submit file
% clear
% clc

% Add path
addpath('../feature extract', '../preprocess','../ReadData3D_version1k/nii','../data');

% Load grey histograms
disp('Load grey histograms into workspace ...')

% Folder
folder_test = '../data/greyhisto_test/';
extension = '*.mat';
listing = dir(strcat(folder_test,extension));
NoB = length(listing);
greyHisto_test = cell(NoB,1);

for i = 1:length(listing)
    file_name = listing(i).name;

    % skip system files (.etc)
    if file_name(1) == '.'
        continue
    end
    
    % find file number (file of the form 'test_3.nii'
    parse1 = strsplit(file_name, '.');
    parse2 = strsplit(parse1{1}, '_');
    file_number = str2num(parse2{2});
    path_name = strcat(folder_test, '/', file_name);
    
    % Load grey histogram
    load(path_name);
    
    % Put into greyHisto
    greyHisto_test{file_number} = temp_GH;
    
end

% Load train grey histograms
% Folder
folder_train = '../data/greyhisto_train/';
% extension = '*.mat';
listing = dir(strcat(folder_train,extension));
NoB = length(listing);
greyHisto_train = cell(NoB,1);

for i = 1:length(listing)
    file_name = listing(i).name;

    % skip system files (.etc)
    if file_name(1) == '.'
        continue
    end
    
    % find file number (file of the form 'test_3.nii'
    parse1 = strsplit(file_name, '.');
    parse2 = strsplit(parse1{1}, '_');
    file_number = str2num(parse2{2});
    path_name = strcat(folder_train, '/', file_name);
    
    % Load grey histogram
    load(path_name);
    
    % Put into greyHisto
    greyHisto_train{file_number} = temp_GH;
    
end

disp('Grey histograms loaded successfully')

% Load betas and parameters
% load('../data/safe_opt4/Struct_getCSF_Tissue_03cubic.mat')

% Showed good RMSE and cvRMSE
% n = 7379;   % Improved score, no norm, only linear
% n = 27292;
% n = 28322; % quad norm, also for not normalized
% n = 47988; % cubic, no norm, super duper, however could be little overfitting the test data
% n = 88924; % cubic, with norm
% n = 5549; %quintic and quartic ... but only for feature 2

% cg1 = Limits_cell{n}(1);
% ra1 = Limits_cell{n}(2);
% cg2 = Limits_cell{n}(3);
% ra2 = Limits_cell{n}(4);
% cg3 = Limits_cell{n}(5);
% ra3 = Limits_cell{n}(6);

% Scored best so far
Limits = [220 50 750 35 1450 110];

cg1 = Limits(1);
ra1 = Limits(2);
cg2 = Limits(3);
ra2 = Limits(4);
cg3 = Limits(5);
ra3 = Limits(6);

parameters = struct('cgone',cg1,'rangeone',ra1,'cgtwo',cg2,'rangetwo',ra2,...
                  'cgthree',cg3,'rangethree',ra3);

% betas = Save_b{n};

% Generate X_test values        
fun = 'getCSF_Tissue_04';
% fun = 'feature_extract_3peaks_v1';
NoF = 3;

% Create model
% greyHisto_train replaced by path name
[betas,~,RMSE,cvRMSE] = train_b_cv(greyHisto_train,'../data/targets.csv',fun,parameters,NoF);

Xtest = generate_X_optver(greyHisto_test,fun,parameters,NoF,'test');
% Xtest = generate_X('../data/set_test/',fun,parameters);

% Compute age estimates, quadratic and linear terms currently
y_hat = [ones(138,1) Xtest Xtest.^2 Xtest.^3]*betas;
% y_hat = [ones(138,1) Xtest Xtest(:,2).^2 Xtest(:,2).^3]*betas;


% Create submission file
% check if there is already a file with name 'submit.csv', if so delete it

file = '../data/submit.csv';
if exist(file, 'file') == 2
    delete(file);
end

data_matrix = [(1:138)' y_hat];

cd('../data')
csvwrite('submit.csv',data_matrix);
cd('../testing')



