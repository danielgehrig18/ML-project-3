% Create csv submit file
% clear
% clc

% New: Implemented for use with lasso optimization results

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

% Create parameter struct
% Limits = [220 50 750 35 1450 110];
% 
% cg1 = Limits(1);
% ra1 = Limits(2);
% cg2 = Limits(3);
% ra2 = Limits(4);
% cg3 = Limits(5);
% ra3 = Limits(6);
%                 
% parameters = struct('cgone',cg1,'rangeone',ra1,'cgtwo',cg2,'rangetwo',ra2,...
%                   'cgthree',cg3,'rangethree',ra3);

parameters = struct('segments',45);

% Generate X_test values        
% fun = 'getCSF_Tissue_03';
% NoF = 3;
fun = 'feature_extract3';
NoF = 45;
y_file = '../data/targets.csv';

% % Create model
% mdl = 'quadratic';
% [betas,statistics] = train_b_cv_v3(greyHisto_train,y_file,fun,parameters,NoF,mdl);

idx1 = statistics.Index1SE;
intercept1 = statistics.Intercept(idx1);

idx2 = statistics.IndexMinMSE;
% [~,idx2] = min(statistics.MSE);
intercept2 = statistics.Intercept(idx2);

Xtest = generate_X_optver(greyHisto_train,fun,parameters,NoF,'train');

% Compute age estimates, quadratic and linear terms currently
Xtest_des = x2fx(Xtest,mdl);
y_hat1 = Xtest_des*betas(:,idx1) + intercept1;
y_hat2 = Xtest_des*betas(:,idx2) + intercept2;

% % Compute root mean square error
y = csvread(y_file);
RMSE1 = sqrt(1/length(y)*sum(abs((y-y_hat1)).^2));
RMSE2 = sqrt(1/length(y)*sum(abs((y-y_hat2)).^2));


%%
% Create submission file
% check if there is already a file with name 'submit.csv', if so delete it
% 
% file = '../data/submit.csv';
% if exist(file, 'file') == 2
%     delete(file);
% end
% 
% data_matrix = [(1:138)' y_hat2];
% 
% cd('../data')
% csvwrite('submit.csv',data_matrix);
% cd('../testing')



