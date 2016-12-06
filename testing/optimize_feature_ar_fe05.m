% Optimize feature

% Tabula rasa
clear
close all
clc

% Add path
addpath('../feature extract', '../preprocess','../ReadData3D_version1k/nii','../data');

% test feature
% fun = 'getCSF_Tissue_03';
fun = 'getCSF_Tissue_03';
% paramteres of fun
% Optimization part 3
cgones = 215:5:225;
cgtwos = 745:5:755;
cgthrees = 1445:5:1455;
% Oneside-range (full range = 2* oneside-range)
raones = 45:5:55;
ratwos = 30:5:40;
rathrees = 105:5:115;

NumOfLoops = length(cgones)*length(raones)*length(cgtwos)*length(ratwos)*...
    length(cgthrees)*length(rathrees);
disp(['Number of loops is: ' num2str(NumOfLoops)])

% Limits = [220 50 750 35 1450 110];
% segments = 45;

% Quadratic, linear and interaction terms
% mdl = 'quadratic';
% % Linear, quadratic, cubic, quartic and quintic terms, no interaction
% mdl = [0 0 0;1 0 0;0 1 0;0 0 1;2 0 0;0 2 0;0 0 2;3 0 0;0 3 0;0 0 3; ...
%     4 0 0;0 4 0; 0 0 4; 5 0 0; 0 5 0; 0 0 5];

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
Save_MSE = zeros(1,NumOfLoops);
% Save_cvRMSE = Save_RMSE;

% Initalize diary log
% diary;

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
%                 
                parameters = struct('cgone',cg1,'rangeone',ra1,'cgtwo',cg2,'rangetwo',ra2,...
                    'cgthree',cg3,'rangethree',ra3);
%                   parameters = struct('segments',segments);
                
                % Number of features to initialize X matrix in
                % generate_X_optver
                NoF = 3;    % Number of regions
%                   NoF = 45;
                % Train model
                % Attention: Changed to !! ------------
                % get_CSF_Tissue_v03 -> normalized
                % quadratic and cubic terms allowed (in train_b_cv)
                
                % Check before running optimization:
                % - feature extractor -> something changed? (maybe
                % suppressed some feature)
                % - train_b_cv -> doing the right thing with the features?
                % - Number of features (above) -> correct number?
                
                [~,MSE] = train_b_cv_v4(greyHisto,y_file,fun,parameters,NoF);
                
%                 Save_X{1,i} = X;
%                 Save_b{1,i} = betas;
                Save_MSE(1,i) = MSE;
%                 Save_cvRMSE(1,i) = cvRMSE;

                % Display results
%                 disp(['Parameters: CG one: ' num2str(cg1) ', Range one: ' num2str(ra1) ...
%                     ', CG two: ' num2str(cg2) ', Range two: ' num2str(ra2) ...
%                     ', CG three: ' num2str(cg3) ', Range three: ' num2str(ra3)]);
                disp(['Mean squared error: ' num2str(MSE)]);
%                 disp(['Cross-validation root mean squared error: ' num2str(cvRMSE)]);
                disp('-----------------------');
end                  

% Get elapsed time
elapsedTime = toc;
% Convert to hours (from seconds, default)
elapsedTimeh = elapsedTime/3600;

disp(['The generation took ' num2str(elapsedTimeh) ' hours or ' num2str(elapsedTime) ...
    ' seconds.']);

% Plot MSE vs. lambda
% lassoPlot(betas,statistics,'PlotType','CV')

% diary off

% % Print best result
% S = Save_RMSE;
% S1 = Save_cvRMSE;
% 
% [m,i] = min(S);
% disp(['Minimum of RMSE: ' num2str(m) ', at ' num2str(i) ' and an cvRMSE of: ' num2str(S1(i)) '.'])
% Limits_cell{i}

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
