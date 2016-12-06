% Optimize feature

% Tabula rasa
clear
close all
clc

% Add path
addpath('../feature extract', '../preprocess','../ReadData3D_version1k/nii','../data');

% test feature
fun = 'feature_extract2';
% paramteres of fun
% Set maximum grey value in gwv_weights_modar: 1750

% Test
% lower_limits = 0:delgrey:20;
% upper_limits = delgrey:delgrey:30;

% Optimization 01 of feature extract2 aborted, too long (> 1.6d)
delgrey = 10;   %grey value step size
lower_limits = 0:delgrey:1750-2*delgrey;
upper_limits = delgrey:delgrey:1750-delgrey;

% % Optimization 02 of feature extract2
% delgrey = 50;   %grey value step size
% lower_limits = 0:delgrey:1750-2*delgrey;
% upper_limits = delgrey:delgrey:1750-delgrey;

lower_limits(1) = 1;
% Minimum window size = 10 (delgrey)

NumOfLoops = sum(1:length(lower_limits));
Limits_lu = cell(1,NumOfLoops);
idx = 0;

for i = 1:length(lower_limits)
    for k = i:length(upper_limits)
        idx = idx +1;
        Limits_lu{1,idx} = [lower_limits(i) upper_limits(k)];
    end
end

% Create diary file name (incl. timestamp)
t = datetime('now');
tsplit = strsplit(char(t),' ');
tsplit2 = strsplit(char(tsplit(2)),':');
timestamp = strcat(tsplit(1),'-',tsplit2(1),'-',tsplit2(2),'-',tsplit2(3));
diaryname = strcat(fun,'_',timestamp);

% Get files
folder = '../data/greyhisto_train/';

% Location of file containing target values y
y_file = '../data/targets.csv';

% Create save arrays
Save_X = cell(1,NumOfLoops);
Save_b = Save_X;
Save_RMSE = zeros(1,NumOfLoops);
Save_cvRMSE = Save_RMSE;

% Initalize diary log
diary;

disp('Load grey histograms into workspace ...')

% Load grey histograms
extension = '*.mat';
listing = dir(strcat(folder,extension));
NoB = length(listing);
greyHisto = cell(NoB,1);

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
    path_name = strcat(folder, '/', file_name);
    
    % Load grey histogram
    load(path_name);
    
    % Put into greyHisto
    greyHisto{file_number} = temp_GH;
end

disp('Grey histograms loaded successfully')

disp(['STARTED OPTIMIZATION 2nd OF ' fun 'at' datestr(t)]);
disp('Maximum grey value is set at: 1750');

tic

parfor i = 1:NumOfLoops                
                % Create parameter struct
                l = Limits_lu{i}(1,1);
                u = Limits_lu{i}(1,2);
                
                parameters = struct('lower_limit',l,'upper_limit',u);
                
                % Number of features to initialize X matrix in
                % generate_X_optver
                NoF = 3;    % Ventricles, Grey Matter, White Matter
                % Train model
                [betas,X,RMSE,cvRMSE] = train_b_cv(greyHisto,y_file,fun,parameters,NoF);
                
                Save_X{1,i} = X;
                Save_b{1,i} = betas;
                Save_RMSE(1,i) = RMSE;
                Save_cvRMSE(1,i) = cvRMSE;

                % Display results
                disp(['Parameters: Lower Limit: ' num2str(l) ' Upper Limit: ' num2str(u)]);
                disp(['Root mean squared error: ' num2str(RMSE)]);
                disp(['Cross-validation root mean squared error: ' num2str(cvRMSE)]);
                disp('-----------------------');
                                    
end

% Get elapsed time
elapsedTime = toc;
% Convert to hours (from seconds, default)
elapsedTimeh = elapsedTime/3600;

disp(['The generation took ' num2str(elapsedTimeh) ' hours or ' num2str(elapsedTime) ...
    ' seconds.']);

% Go to safe_opt folder in data
cd('../data/safe_opt2')
save('Struct_feature_extract2run2.mat','diaryname','Limits_lu','Save_X','Save_b','Save_RMSE','Save_cvRMSE');
% Go back to testing folder
cd('../../testing')

diary off
