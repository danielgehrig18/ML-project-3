
% add relevant folder to path
%addpath('feature extract', 'Source','ReadData3D_version1k/nii');

%% choose function and its parameters
fun = 'MLP2_feature_extract1';
parameters = struct('x_segments', 3, ...
                    'y_segments', 3, ...
                    'z_segments', 3, ...
                    'bins', 40);

% train model with Matlab function LinearModel.fit
%X = generate_X('../data/set_train', fun, parameters);
y = csvread('../targets.csv');

fun = str2func('crossvalidation');
inmodel = sequentialfs(fun, X,y,  'direction', 'backward', 'options', struct('Display', 'iter'));