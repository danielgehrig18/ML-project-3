function [ Histobounds ] = MLP2_getHistobounds( Xint )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
% Compute histogram bounds for each chunk
% Load or compute minimum and maximum intensities

[~,yi] = size(Xint);
Histobounds = zeros(yi/2,2);

for i = 1:yi/2
    % Use maximum and minimum of all maxima and minima as global histogram
    % bounds
%     Histobounds(i,1) = min(Xint(:,(i-1)*2+1));
%     Histobounds(i,2) = max(Xint(:,(i-1)*2+2));
    % Use median of all maxima and minima as global histogram bounds
    Histobounds(i,1) = round(median(Xint(:,(i-1)*2+1)),0);
    Histobounds(i,2) = round(median(Xint(:,(i-1)*2+2)),0);
end

% figure(1)
% scatter(1:yi/2,Histobounds(:,1))
% figure(2)
% scatter(1:yi/2,Histobounds(:,2))

end

