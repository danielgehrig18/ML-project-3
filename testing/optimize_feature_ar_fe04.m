% Optimize feature

% Tabula rasa
clear
close all
clc

% Add path
addpath('../feature extract', '../preprocess','../ReadData3D_version1k/nii','../data');

% test feature
fun = 'getCSF_Tissue_03';
% paramteres of fun
% First optimization
cgones = 180:20:260;
cgtwos = 700:50:900;
cgthrees = 1300:50:1500;
% Oneside-range (full range = 2* oneside-range)
raones = 10:10:70;
ratwos = 10:25:210;
rathrees = 10:25:310;

NumOfLoops = length(cgones)*length(raones)*length(cgtwos)*length(ratwos)*...
    length(cgthrees)*length(rathrees);

Limits_cell = cell(NumOfLoops,1);
count = 0;
% Initialize limits cell
for co = cgones
    for ct = cgtwos
        for cth = cgthrees
            for ro = raones
                for rt = ratwos
                    for rth = rathrees
                        count = count+1;
                        Limits_cell{count} = [co ro ct rt cth rth];
                    end
                end
            end
        end
    end
end


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
% Save_X = cell(1,NumOfLoops);
% Save_b = Save_X;
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
                cg1 = Limits_cell{i}(1);
                ra1 = Limits_cell{i}(2);
                cg2 = Limits_cell{i}(3);
                ra2 = Limits_cell{i}(4);
                cg3 = Limits_cell{i}(5);
                ra3 = Limits_cell{i}(6);
                
                parameters = struct('cgone',cg1,'rangeone',ra1,'cgtwo',cg2,'rangetwo',ra2,...
                    'cgthree',cg3,'rangethree',ra3);
                
                % Number of features to initialize X matrix in
                % generate_X_optver
                NoF = 3;    % Number of regions/features
                % Train model
                               
                % Check before running optimization:
                % - feature extractor -> something changed? (maybe
                % suppressed some feature)
                % - train_b_cv -> doing the right thing with the features?
                % - Number of features (above) -> correct number?
                
                [~,~,RMSE,cvRMSE] = train_b_cv(greyHisto,y_file,fun,parameters,NoF);
                
                
                Save_RMSE(1,i) = RMSE;
                Save_cvRMSE(1,i) = cvRMSE;

                % Display results
                disp(['Parameters: CG one: ' num2str(cg1) ', Range one: ' num2str(ra1) ...
                    ', CG two: ' num2str(cg2) ', Range two: ' num2str(ra2) ...
                    ', CG three: ' num2str(cg3) ', Range three: ' num2str(ra3)]);
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

diary off

% Print best result
S = Save_RMSE;
S1 = Save_cvRMSE;

[m,i] = min(S);
disp(['Minimum of RMSE: ' num2str(m) ', at ' num2str(i) ' and an cvRMSE of: ' num2str(S1(i)) '.'])

% idx1 = find(S < 7.9 & S1<7.9 & S < S1);

% % Go to safe_opt folder in data
% cd('../data/safe_opt4')
% % save('Struct_getCSF_Tissue_03cubic_run4.mat','diaryname',...
% %     'Limits_cell','Save_X','Save_b','Save_RMSE','Save_cvRMSE');
% 
% % New: Features and Model is not saved (use less memory)
% save('Struct_getCSF_Tissue_03quartic_single.mat','diaryname',...
%     'Limits_cell','Save_RMSE','Save_cvRMSE');
% % Go back to testing folder
% cd('../../testing')
