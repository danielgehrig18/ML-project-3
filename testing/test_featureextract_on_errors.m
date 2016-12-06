% Test feature extract function

% add relevant folder to path
addpath('../feature extract', '../preprocess','../ReadData3D_version1k/nii',...
    '../data/set_train');

% List files
folder = '../data/set_train/';
extension = '*.nii';
files = dir(char(strcat(folder,extension)));

path_name = char(strcat(folder,files(1).name));

% Create parameters struct
h = 10;
p = 10;
k = 10;

parameters = struct('x_segments',h,'y_segments',p,'z_segments',k);

feature = feature_extract1(path_name, parameters);
