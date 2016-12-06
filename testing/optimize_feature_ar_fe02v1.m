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
delgrey = 50;   %grey value step size

% Test
% lower_limits = 0:delgrey:20;
% upper_limits = delgrey:delgrey:30;

% Optimization 01 of feature extract2 aborted, too long (> 1.6d)
% lower_limits = 0:delgrey:1750-2*delgrey;
% upper_limits = delgrey:delgrey:1750-delgrey;

% Optimization 02 of feature extract2
lower_limits = 0:delgrey:1750-2*delgrey;
upper_limits = delgrey:delgrey:1750-delgrey;

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

% Initalize diary log
diary;

disp(['STARTED OPTIMIZATION OF ' fun 'at']);
disp('Maximum grey value is set at: 1750');

% Get files
path_name = '../data/set_train/';

% Location of file containing target values y
y_file = '../data/targets.csv';

tic

% Create save arrays
Save_X = cell(1,NumOfLoops);
Save_b = Save_X;
Save_RMSE = zeros(1,NumOfLoops);
Save_cvRMSE = Save_RMSE;

parfor i = 1:NumOfLoops                
                % Create parameter struct
                l = Limits_lu{i}(1,1);
                u = Limits_lu{i}(1,2);
                
                parameters = struct('lower_limit',l,'upper_limit',u);
                
                % Number of features to initialize X matrix in
                % generate_X_optver
                NoF = 3;    % Ventricles, Grey Matter, White Matter
                % Train model
                [betas,X,RMSE,cvRMSE] = train_b_cv(path_name,y_file,fun,parameters,NoF);
                
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
elapsedTime = elapsedTime/3600;

disp(['The optimization took ' num2str(elapsedTime) ' hours.']);

% Go to safe_opt folder in data
cd('../data/safe_opt2')
save('Struct_feature_extract2.mat','diaryname','Limits_lu','Save_X','Save_b','Save_RMSE','Save_cvRMSE');
% Go back to testing folder
cd('../../testing')

diary off

% %%
% % Plot results in 3d plot
% X1 = zeros(NumOfLoops,1);
% Y1 = X1;
%    
% for i = 1:NumOfLoops
%     X1(i,1) = Limits_lu{i}(1,1);
%     Y1(i,1) = Limits_lu{i}(1,2);
% end
% 
% Z1 = Save_RMSE;
% Z2 = Save_cvRMSE;
% 
% close
% figure(1)
% scatter3(X1,Y1,Z1)
% hold on
% scatter3(X1,Y1,Z2)
% hold off
% xlabel('Lower Limit')
% ylabel('Upper Limit')
% zlabel('RMSE')
% legend('RMSE','cvRMSE')
% title('Results of optimization, step size = 50')
% 
% figure(2)
% subplot(2,1,1)
% scatter(X1,Z1)
% hold on
% scatter(X1,Z2)
% hold off
% xlabel('Lower Limit')
% ylabel('RMSE')
% 
% subplot(2,1,2)
% scatter(Y1,Z1)
% hold on
% scatter(Y1,Z2)
% hold off
% xlabel('Upper Limit')
% ylabel('RMSE')




