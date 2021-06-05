%%%%%%%%%%%% test code for standalone WavResNet and sequential WavResNet%%%
%|- for standalone WavResNet: num_epoch=200, Layer=1, wrn_stand=1
%|- for sequential WavResNet: num_epoch=50, Layer=15, wrn_stand=0
%| Ye siqi, UM-SJTU JI, Shanghai Jiao Tong Univ.
%| 2020-05

clear all;
close all;
%addpath('~/exportfig');
%% Path setting
 run '../irt/setup.m';
gpuIdx = gpuDevice(1);

run('~/matconvnet-1.0-beta24/matlab/vl_setupnn.m'); % MatConvNet path
 addpath(genpath('~/SUPER-for-CT-recon/'));
%addpath(genpath('../toolbox/'));
%addpath(genpath('./lib_contourlet/'));
wrn_stand = 1; % standalone or sequential
if wrn_stand == 1
 netpath = 'trained_model/wavresnet_standalone/';
% savepath = 'pureWRN_mm2HU/nufft_6pat_lr-5_imgs500_PatSz256_Overlap10_BatSz1_WgtDecay1e-02_GradMax1e-03';
savepath = 'wavresnet/';
else
 netpath = 'trained_model/wavresnet_sequential/';
% savepath = 'pureWRN_rnn/preLyrInit_nufft_6pat_lr-5_imgs500_PatSz256_Overlap10_BatSz1_WgtDecay1e-02_GradMax1e-03';
savepath = 'wrn_seq/';
end
if ~exist(savepath,'dir') mkdir(savepath); end
%% target geometry for compute RMSE and SSIM
down = 1; % downsample rate
sg = sino_geom('fan', 'units', 'mm', 'nb',736, 'na',1152,'orbit',360, 'ds',1.2858,...
     'strip_width','ds','dsd',1085.6,'dso',595,'dfs',0, 'down', down);

ig = image_geom('nx',512,'fov',sg.rfov*sqrt(2)); % dx=0.69298mm, fov=354.8065mmi
SqrtPixNum = sqrt(sum(ig.mask(:)>0));
%% Parameters
lv                  = [1,2,3];              % vector of numbers of directional filter bank decomposition levels at each pyramidal level
dflt                = 'vk';                 % filter name for the directional decomposition step
patchsize           = 256; %30;                   % the size of patch
batchsize           = 1;                   % the size of batch
wgt                 = 1e3;                  % weight multiplied to input
num_epoch           = 200; %50; % 120;                  % the number of epochs
lr_rate             = [-2 -5];% [-2 -5];              % learing rate scheduling from 1e-2 to 1e-5
num_lr_epoch        = num_epoch;            
wgtdecay            = 1e-2;                 % weight decay parameter
gradMax             = 1e-3;                 % gradient clipping

gpus                = 1;                    % gpu on / off
train               = struct('gpus', gpus);

% expdir              = 'training_1';         % experiment name
% if ~exist(expdir, 'dir'), mkdir(expdir); end;
method = 'residual';
Layer = 1; % 15;
% lambda = 1 ; %0.5 for parallel super
overlap = 10; %5;
%% Denoising module: wavresnet or fbpconvnet
hu2inten = @(x) x./(1000/0.0192);
inten2hu = @(x) x.*(1000/0.0192);

printm('load testing data ... \n');
test_slice = [20,50,100,150,200];
% slice = 100;
printm('load testing data ... \n');
caselist = {'L067','L143','L192','L310'};
for ilist = 1:length(caselist) % 7:10; %1:3
        study = caselist{ilist};
        load(['~/Desktop/MayoData_gen/' study  '/full_3mm_img.mat']);
        load(['~/Desktop/MayoData_gen/' study  '/sim_low_nufft_1e4/xfbp.mat']);

    for ii =1:length(test_slice)
            xfbp_low = xfbp(:,:,test_slice(ii));
            xref = xfdk(:,:,test_slice(ii));
            xrla = xfbp_low;
            xrla_msk = xrla(ig.mask);
            xref_val = xref;
            info_val.RMSE(1) = norm(xrla_msk - xref(ig.mask))/SqrtPixNum;
            info_val.SSIM(1) = ssim(xrla,xref);
            info_val.PSNR(1) = computeRegressedSNR(xrla_msk,xref(ig.mask));
    % end
     xfbp_low = hu2inten(xfbp_low);


    for ilayer = 1:Layer
	if wrn_stand ==1
        load ([netpath sprintf('/net-epoch-%d.mat',num_epoch)]);
	else
        load ([netpath sprintf('/net-Epoch%d-layer-%d.mat',num_epoch,ilayer)]);
       end
	 net = vl_simplenn_move(net,'gpu');
        xini = cnn_CT_denoising_forward_process(net,xfbp_low,lv,dflt,patchsize,batchsize,overlap,wgt,gpus);
        xrla = inten2hu(xini);

        xrla_msk = xrla(ig.mask);
        info_val.xx(:,:,ilayer) = xrla;
        info_val.RMSE(ilayer+1) = norm(xrla_msk - xref(ig.mask))/SqrtPixNum;
        info_val.SSIM(ilayer+1) = ssim(xrla,xref);
        info_val.PSNR(ilayer+1) = computeRegressedSNR(xrla_msk,xref(ig.mask));
        figure(1);
        imshow(info_val.xx(:,:,ilayer),[800 1200]);
        %export_fig([savepath '/test_1e4/' study sprintf('s%d_epoch%d_ilayer%d.pdf',test_slice(ii),num_epoch,ilayer)],'-transparent');
%         xfbp_low = lambda * xini + (1-lambda) * xfbp_low ;
        xfbp_low = hu2inten(xrla);

    end
    save([savepath '/test_1e4/' study sprintf('s%d_epoch%d_Layer%d.mat',test_slice(ii),num_epoch,Layer)],'info_val');
	clear info_val
    end

end
