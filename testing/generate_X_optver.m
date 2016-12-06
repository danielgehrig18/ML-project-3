function [ X ] = generate_X_optver(folder, fun, parameters, NumOfFeatures,testortrain)
%GENERATE_X Generate the data matrix from data in folder. Parameters limit1
%and limit2 are for the feature_extract function. 
%Resulting X matrix is of size #data_points x (#features + 1)
% 278 train images
switch testortrain
    case 'train'
        X = zeros(278,NumOfFeatures);
    case 'test'
        X = zeros(138,NumOfFeatures);
end

if ischar(folder)
    % if path is handed over do this
    
    files = dir(folder);
    
    % iterate through files in folder
    for file = files'
        file_name = file.name;

        % skip system files (.etc)
        if file_name(1) == '.'
            continue
        end
    
        % find file number (file of the form 'test_3.nii'
        parse1 = strsplit(file_name, '.');
        parse2 = strsplit(parse1{1}, '_');
        file_number = str2num(parse2{2});
        
        % load file
        path_name = strcat(folder, '/', file_name);
    
        % extract features from file
        f = str2func(fun);
        x = f(path_name, parameters);
        X(file_number, :) = x;
    end
elseif iscell(folder)
    % if cell array containing grey histograms is handed over do this:
    % (array is already sorted from brain 1 to 278 (preprocessed))
    for i = 1:size(X,1)
        f = str2func(fun);
        GH = folder{i};
        x = f(GH,parameters);
        X(i,:) = x;
    end
end


end

