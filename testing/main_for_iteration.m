% main - main pipeline for training b with a training set from
% 'data/set_train' folder and submission of test values from
% 'data/set_train folder. Train_b is optimized for best r_squared
% (coefficient of determination).

% add relevant folder to path
addpath('feature extract', 'preprocess','ReadData3D_version1k/nii');
iteration1 = 0;
iteration2 = 0;
bestparams = [];
besterror = 999;

iteration_1_lower = 200;
iteration_1_upper = 500;
iteration_1_step = 20;

iteration_2_lower = 20;
iteration_2_upper = 200;
iteration_2_step = 20;

for iteration1 = iteration_1_lower:iteration_1_step:iteration_1_upper
    for iteration2 = iteration_2_lower:iteration_2_step:iteration_2_upper
%% choose function1 and its parameters
        fun1 = 'feature_extract_locations';
        parameters1 = struct('up', 1,...
                            'down',1,...
                            'centerpoint', 0, ...
                            'offset', iteration2,...
                            'target', iteration1);
        [model1, X1] = train_b('data/set_train', 'data/targets.csv', fun1, parameters1);

        Y = csvread('data/targets.csv');
        X1 = orderfeatures(X1);
        plot(X1)
        disp(['The RMSE is: ' num2str(model1.RMSE) ' for value1: ' num2str(iteration1)...
            ' value2: ' num2str(iteration2) ' - ' num2str(100*(iteration1 - iteration_1_lower) /...
            (iteration_1_upper - iteration_1_lower)) ' % done!']);
        if (model1.RMSE < besterror)
            besterror = model1.RMSE;
            bestparams = [iteration1, iteration2];
        end
    end
end
disp(['Optimization terminated with best error: ' num2str(besterror)...
    ' @vlaue1: ' num2str(bestparams(1,1)) ' and value2: ' num2str(bestparams(1,2))]);
