%%%%%%%%% standalone WavResNet or sequential WavResNet %%%%%%%%%
%| parameter settings reported in the paper:
%|- for standalone WavResNet: num_epoch=200, Layer=1
%|- for sequential WavResNet: num_epoch=50, Layer=15
%| Ye siqi, UM-SJTU JI, Shanghai Jiao Tong Univ.
%| 2020-05
clear all;
close all;

%% Path setting
run '../irt/setup.m';
 gpuIdx = gpuDevice(1);

run('~/matconvnet-1.0-beta24/matlab/vl_setupnn.m'); % MatConvNet path
 addpath(genpath('~/SUPER-for-CT-recon/'));
%addpath(genpath('../toolbox/'));
%addpath(genpath('./lib_contourlet/'));
%% target geometry for compute RMSE and SSIM
down = 1; % downsample rate
sg = sino_geom('fan', 'units', 'mm', 'nb',736, 'na',1152,'orbit',360, 'ds',1.2858,...
     'strip_width','ds','dsd',1085.6,'dso',595,'dfs',0, 'down', down);

ig = image_geom('nx',512,'fov',sg.rfov*sqrt(2)); % dx=0.69298mm, fov=354.8065mm
SqrtPixNum = sqrt(sum(ig.mask(:)>0));
%% Parameters
lv                  = [1,2,3];              % vector of numbers of directional filter bank decomposition levels at each pyramidal level
dflt                = 'vk';                 % filter name for the directional decomposition step
patchsize           = 256;                   % the size of patch
batchsize           = 1; % 10;                   % the size of batch
wgt                 = 1e3;                  % weight multiplied to input
num_epoch           = 50;                  % the number of epochs
lr_rate             = [-2 -5]; %[-2 -7];              % learing rate scheduling from 1e-2 to 1e-5
num_lr_epoch        = num_epoch;            
wgtdecay            = 1e-2;                 % weight decay parameter
gradMax             = 1e-3;                 % gradient clipping

gpus                = 1;                    % gpu on / off
train               = struct('gpus', gpus);

% expdir              = 'training_1';         % experiment name
% if ~exist(expdir, 'dir'), mkdir(expdir); end;
method = 'residual';
Layer = 15;
layer_start =1;
overlap = 10;


%% Denoising module: wavresnet or fbpconvnet
printm('load training data ... \n');
load('/home/share/data_train500_val20.mat');
num_train = 500; num_val = 20;
xfbp_val = xfbp_low_val;
xfbp_train = xfbp_low_train;
clear xfbp_low_val xfbp_low_train;
xref = cat(3,xref_train,xref_val);
xfbp_low = cat(3,xfbp_train,xfbp_val); 
%%
hu2inten = @(x) x./(1000/0.0192);
inten2hu = @(x) x.*(1000/0.0192);
xref_inten = hu2inten(xref);
% kappa_low = kappa_low(:,:,1:train_slice);

printm 'wavCoef of xref...'
clear mex;
xref_coef = single(cnn_wavelet_decon(double(xref_inten),lv,dflt));
printm 'xref decompose finished!'
%whos xref_coef
imdb.images.label = xref_coef;
imdb.images.set = [ones(num_train,1);2*ones(num_val,1)];
imdb.meta.sets = {'train' 'val' 'test'};
clear xref_coef xref_inten;

wavdecon = tic;
clear mex;
printm 'wavCoef of xfbp...'
xfbp_low = hu2inten(xfbp_low);
xfbp_coef = single(cnn_wavelet_decon(double(xfbp_low),lv,dflt));
printm 'xfbp decompose finished!'

imdb.images.data = xfbp_coef;
tdecon = toc(wavdecon)
clear xfbp_coef;
expdir = sprintf('pureWRN_rnn/preLyrInit_nufft_6pat_lr-5_imgs%d_PatSz%d_Overlap%d_BatSz%d_WgtDecay%.0e_GradMax%.0e/',num_train,patchsize,overlap,batchsize,wgtdecay,gradMax);    % experiment name
if ~exist(expdir, 'dir'), mkdir(expdir); end

printm 'start training ...'
ttrain = tic;

[net, info] = cnn_CT_denoisingi_rnn_prevLyrInit(imdb,overlap,layer_start,Layer, ...
    'expDir',       expdir,     'method',  method,             ...
    'numEpochs',    num_epoch,     ...
    'patchSize',    patchsize,  'batchSize',    batchsize,              ...
    'wgt',          wgt,        'lrnrate',      lr_rate,                ...
    'wgtdecay',     wgtdecay,   ...
    'gradMax',      gradMax,    'num_lr_epoch', num_lr_epoch,           ...
    'train',        train);
ttrain = toc(ttrain)


