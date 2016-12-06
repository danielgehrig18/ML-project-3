function [ X ] = generate_X(folder, fun, parameters)
%GENERATE_X Generate the data matrix from data in folder. Parameters limit1
%and limit2 are for the feature_extract function. 
%Resulting X matrix is of size #data_points x (#features + 1)
files = dir(folder);


h = waitbar(0,'Evaluating...');
count = 0;
totcount = length(files)-2;

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
    
    % Show waitbar
    count = count+1;
    waitbar(count/totcount);
end
close(h)
end

