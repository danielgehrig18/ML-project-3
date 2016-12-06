% Show grey histograms

% Load certain histograms into workspace
folder = '../data/greyhisto_train/';
% Which histograms are loaded (Number)

% 

ages = {'20','21','40','41','65','60','85','82','87','93'};
loadHistogram = [1 20 53 62 78 92 143 144 149 160];
% loadHistogram = 1;

% ages = {'87','89','89','90','96','94'};
% loadHistogram = [31 99 89 118 168 114];
% Handle cell
H = cell(1,length(loadHistogram));
% Handle cell smoothed
H_s = H;
% Span, choose odd number
p = 9;
count = 0;
for i = loadHistogram
    count = count+1;
    load(strcat(folder,'train_',num2str(i),'.mat'));
    H{count} = temp_GH;
    % Smoothing with moving average with span p
    H_s{count} = smooth(temp_GH,p);
end

% H2 = feature_extract10('../data/set_train/train_2.nii',5000);

%%
% Filter grey value histogram (get rid of the noise -> Differantiation)
% p2 = 8;
% p3 = 20;
% H_smooth = medfilt1(H{1},p3);
% H_smooth2 = smooth(H{1},p2,'rloess');
% H_smooth3 = smooth(H{1},p3); % led to best results

%%
% Scale grey values w.r.t. the largest grey value of all loaded images
% Find largest grey value
maxval = 0;
for i = 1:length(loadHistogram)
    if maxval < size(H{i},1)
        maxval = size(H{i},1);
    end
end

% xH2 = 1:length(H2);

% plot
close
for i = 1:length(loadHistogram)
%     X = normhisto(H{i})*maxval;
    plot(H_s{i})
    hold on
end
% plot(xH2+10,H2)
legend(ages)
xlabel('Grey values')
ylabel('Quantity')
title('Smoothed grey value histograms')
xlim([0 2000])
hold off

%% Error
E1 = sum(abs(H_smooth - H{1}).^2);
E2 = sum(abs(H_smooth2-H{1}).^2);
E3 = sum(abs(H_smooth3-H{1}).^2);

