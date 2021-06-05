% training FBPConvNet
% modified from MatconvNet (ver.23)

clear; clc
reset(gpuDevice)
%restoredefaultpath
%addpath(genpath('/n/ludington/x/ysiqi/MayoData_3mm/denoise_recon/fbpconvnet'))
run('/home/matconvnet-1.0-beta24/matlab/vl_setupnn.m');


%%%%%%%%%%%%%%% load training data here %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

W   = 512; % size of patch
Nimg= size(lab_n,3); % # of train + test set
Nimg_test= 20;
%Nimg_test= fix(Nimg*0.05);

lab_n = reshape(lab_n, W, W, [], Nimg);
lab_d = reshape(lab_d, W, W, [], Nimg);

train_opts.channel_in = 1;
train_opts.channel_out= 1;

id_tmp  = ones(Nimg,1);
id_tmp(Nimg-Nimg_test+1:end)=2;

imdb.images.set=id_tmp;             % train set : 1 , test set : 2
imdb.images.noisy=single(lab_d);    % input  : H x W x C x N (X,Y,channel,batch)
imdb.images.orig=single(lab_n);     % output : H x W x C x N (X,Y,channel,batch)

%% 
opt='none';
train_opts.useGpu = 'true'; %'false'
train_opts.gpus = 1 ;       % []
train_opts.patchSize = 512;
train_opts.batchSize = 1;
train_opts.gradMax = 1e-2;  % 1e-2
train_opts.numEpochs = 1001;
train_opts.momentum = 0.99 ;
train_opts.imdb=imdb;
train_opts.expDir = fullfile('./training_result',[num2str(date) '_dose1e5_fullfbp_Epoch501_'],[opt '_x' num2str(dsr)] ,'/');

[net, info] = cnn_fbpconvnet(train_opts);

