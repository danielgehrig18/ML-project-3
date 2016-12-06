% Optimize feature

% Tabula rasa
clear
close all
clc

% Add path
addpath('../feature extract', '../preprocess','../ReadData3D_version1k/nii','../data');

% test feature
fun = 'feature_extract1';
% paramteres of fun
x_segments = 2:1:10;
y_segments = 2:1:10;
z_segments = 2:1:10;

% % Create save struct
% f1 = 'Parameters';
% v1 = {};
% f2 = 'RMSE';
% v2 = {};
% f3 = 'cvRMSE';
% v3 = {};
% save_struct = struct(f1,v1,f2,v2,f3,v3);

% Create diary file name (incl. timestamp)
t = datetime('now');
tsplit = strsplit(char(t),' ');
tsplit2 = strsplit(char(tsplit(2)),':');
timestamp = strcat(tsplit(1),'-',tsplit2(1),'-',tsplit2(2),'-',tsplit2(3));
diaryname = strcat(fun,'_',timestamp);

% Initalize diary log
diary(char(diaryname));

fprintf('STARTED OPTIMIZATION OF "%s" at %s. \n',fun,t);

% Get files
path_name = '../data/set_train/';

% Location of file containing target values y
y_file = '../data/targets.csv';
% 
% % Initialize 
% RMSEonly_old = 999999999999999;
% RMSEcomp_old = RMSEonly_old;
% cvRMSEonly_old = RMSEonly_old;
% cvRMSEcomp_old = RMSEonly_old;
% 
% tol = 0.2;

% Create save vector for RMSE and cvRMSE
safe_bothRMSE = zeros(length(x_segments)*length(y_segments)*length(z_segments),2);

tic
count = 0;
for h = x_segments
        for p = y_segments
            for k = z_segments
                count = count+1;
                % Create parameter struct
                parameters = struct('x_segments',h,'y_segments',p,'z_segments',k);
                % Number of features to initialize X matrix in
                % generate_X_optver
                NoF = h*p*k;
                % Train model
                [betas,X,RMSE,cvRMSE] = train_b_cv(path_name,y_file,fun,parameters,NoF);
                
                safe_bothRMSE(count,:) = [RMSE cvRMSE];
                
                % Go to safe_opt folder in data
                cd('../data/safe_opt1')
                % Save obtained data
                safe_name = char(strcat(num2str(h),num2str(p),num2str(k),'_',fun,'.mat'));
                save(safe_name,'betas','X','RMSE','cvRMSE');
                % Go back to testing folder
                cd('../../testing')
                
                % Display results
                fprintf('Parameters: X Segments: %d, Y Segments: %d, Z Segments: %d \n',h,p,k);
                fprintf('Root mean squared error: %d \n', RMSE);
                fprintf('Cross-validation root mean squared error: %d \n', cvRMSE);
                fprintf('----------------------- \n');
                                    
            
            end
        end
end

% Get elapsed time
elapsedTime = toc;
% Convert to hours (from seconds, default)
elapsedTime = elapsedTime/3600;

fprintf('The optimization took %d hours \n', elapsedTime);

