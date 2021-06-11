% Read fdk reconstruced image data and (post-log) projection data
% Codes refered from Rui Liu
% Siqi Ye, UM-SJTU Joint Institute
clear; close all
dir = '/home/MayoData/'; % change to your Mayo Data folder
study = 'L067'; % patient name

ImagePath = [dir study '/full_3mm/']; % we use 3mm thickness image data

s = dir([ImagePath, '*.IMA']);
allNames = {s.name};
fileNum = max(size(allNames));
parfor k = 1:1:fileNum
    str = [char(ImagePath),char(allNames(1,k))]; 
    xfdk(:,:,1,k) = dicomread(str);
end
xfdk = reshape(single(xfdk),[512 512 fileNum]);
figure;im('mid3',permute(xfdk,[2 1 3]),[]);
save(['data/' study '/full_3mm_img.mat'],'xfdk');
% % unit of readout fdk images: (shifted-)HU



