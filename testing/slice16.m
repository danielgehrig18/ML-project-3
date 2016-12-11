function [ block16 ] = slice16( im, i, axdir,th )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

[x,y,z] = size(im);
slices = 16;
block16 = cell(1,slices);
lev = 2;

switch axdir
    case 'x'
        NoS = floor(x/slices);
        for p = 0:slices-1
            rgbim = squeeze(im(i+p*NoS,:,:));
%             [rgbim, ~] = quantgray(squeeze(im(i+p*NoS,:,:)),lev);
%             rgbim = edge(squeeze(im(i+p*NoS,:,:)),'canny',th);
            block16{p+1} = rgbim;
        end
    case 'y'
        NoS = floor(y/slices);
        for p = 0:slices-1
            [rgbim, ~] = quantgray(squeeze(im(:,i+p*NoS,:)),lev);
%             rgbim = edge(squeeze(im(:,i+p*NoS,:)),'canny',th);
            block16{p+1} = rgbim;
        end
    case 'z'
        NoS = floor(z/slices);
        for p = 0:slices-1
            rgbim = squeeze(im(:,:,i+p*NoS));
%             [rgbim, ~] = quantgray(squeeze(im(:,:,i+p*NoS)),lev);
%             rgbim = edge(squeeze(im(:,:,i+p*NoS)),'prewitt');
            block16{p+1} = rgbim;
        end
end

block16 = reshape(block16,[4,4]);
block16 = block16';
block16 = cell2mat(block16);

end

