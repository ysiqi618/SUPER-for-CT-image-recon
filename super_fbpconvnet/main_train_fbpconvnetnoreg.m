%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This is the code for training FBPConvNet+DataTerm Only.
% Install MatConvNet in advance
%
% modified from Kyong Jin's FBPConvNet training code
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; clc
reset(gpuDevice(1))
restoredefaultpath

run('~/Desktop/irt/setup.m');
run('/home/matconvnet-1.0-beta24/matlab/vl_setupnn.m');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% load normal dose and low dose images for training and 
% validation processes. We used 500 slices for training and 
% 20 slices for validation.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xfbp_low = cat(3,xfbp_low_train, xfbp_low_val);
xref = cat(3, xref_train, xref_val);

lab_n = xref;
lab_d = xfbp_low;
dsr = 520;

W   = 512; % size of patch
Nimg= size(lab_n,3); % # of train + test set
Nimg_test= 20;

lab_n = reshape(lab_n, W, W, [], Nimg);
lab_d = reshape(lab_d, W, W, [], Nimg);

train_opts.channel_in = 1;
train_opts.channel_out=1;

id_tmp  = ones(Nimg,1);
id_tmp(Nimg-Nimg_test+1:end)=2;

imdb.images.set=id_tmp;             % train set : 1 , test set : 2
imdb.images.noisy=single(lab_d);    % input  : H x W x C x N (X,Y,channel,batch)
imdb.images.orig=single(lab_n);     % output : H x W x C x N (X,Y,channel,batch)

beta = 0; delta = 20; mu_para = 0;
block = 1; Outer = 5; numepoch = 4;
katype = 1;
%% 
opt='none';
train_opts.useGpu = 'true'; %'false'
train_opts.gpus = 1;       % []
train_opts.patchSize = 512;
train_opts.batchSize = 1;
train_opts.gradMax = 1e-2;  % 1e-2
train_opts.numEpochs = 15;
train_opts.momentum = 0.99;
train_opts.imdb=imdb;
train_opts.expDir = fullfile('./training_result',[num2str(date) '_FBPConvNetnoreg_nufft_500slices_mu0_Outer5_epoch4_'],[opt '_x' num2str(dsr)] ,'/');

[net, info] = cnn_fbpconvnetnoreg(beta, delta, mu_para, block, Outer, numepoch, katype,train_opts);


