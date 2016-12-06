% Optimize feature

% Tabula rasa
clear
close all
clc

% Add path
addpath('../feature extract', '../preprocess','../ReadData3D_version1k/nii','../data');

% test feature
fun = 'feature_extract3';
% paramteres of fun
segments = 1:300;

NumOfLoops = length(segments);

% Create diary file name (incl. timestamp)
t = datetime('now');
tsplit = strsplit(char(t),' ');
tsplit2 = strsplit(char(tsplit(2)),':');
timestamp = strcat(tsplit(1),'-',tsplit2(1),'-',tsplit2(2),'-',tsplit2(3));
diaryname = strcat(fun,'_',timestamp);

% Get files
% path_name = '../data/set_train/';
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

disp(['STARTED OPTIMIZATION OF ' fun ' at ' datestr(t)]);

tic

parfor i = 1:NumOfLoops             
                % Create parameter struct
                s = segments(i);
                
                parameters = struct('segments',s);
                
                % Number of features to initialize X matrix in
                % generate_X_optver
                NoF = s;    % Number of segments
                % Train model
                [betas,X,RMSE,cvRMSE] = train_b_cv(greyHisto,y_file,fun,parameters,NoF);
                
                Save_X{1,i} = X;
                Save_b{1,i} = betas;
                Save_RMSE(1,i) = RMSE;
                Save_cvRMSE(1,i) = cvRMSE;

                % Display results
                disp(['Parameters: Segments: ' num2str(s)]);
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
cd('../data/safe_opt3')
save('Struct_feature_extract3.mat','diaryname','segments','Save_X','Save_b','Save_RMSE','Save_cvRMSE');
% Go back to testing folder
cd('../../testing')

diary off



