%%%%%%%%%%%%%% test code for SUPER-WRN-ULTRA %%%%%%%%%
%| 2021-05, Ye Siqi, um-sjtu ji
clear all;
close all;

%% Path setting
 run '../irt/setup.m';
% gpuIdx = gpuDevice(2);

run('~/matconvnet-1.0-beta24/matlab/vl_setupnn.m'); % MatConvNet path
 addpath(genpath('~/SUPER-for-CT-recon/'));
%addpath(genpath('../toolbox/'));
%addpath(genpath('./lib_contourlet/'));
% addpath(genpath('~/exportfig'));
%% target geometry for compute RMSE and SSIM
down = 1; % downsample rate
sg = sino_geom('fan', 'units', 'mm', 'nb',736, 'na',1152,'orbit',360, 'ds',1.2858,...
     'strip_width','ds','dsd',1085.6,'dso',595,'dfs',0, 'down', down);

ig = image_geom('nx',512,'fov',sg.rfov*sqrt(2)); % dx=0.69298mm, fov=354.8065mmi
SqrtPixNum = sqrt(sum(ig.mask(:)>0));
A = Gtomo_nufft_new(sg, ig); 
%% Parameters
lv                  = [1,2,3];              % vector of numbers of directional filter bank decomposition levels at each pyramidal level
dflt                = 'vk';                 % filter name for the directional decomposition step
patchsize           = 256; %30;                   % the size of patch
batchsize           = 1;                   % the size of batch
wgt                 = 1e3;                  % weight multiplied to input
num_epoch           = 50; % 120;                  % the number of epochs
lr_rate             = [-2 -5];% [-2 -5];              % learing rate scheduling from 1e-2 to 1e-5
num_lr_epoch        = num_epoch;            
wgtdecay            = 1e-2;                 % weight decay parameter
gradMax             = 1e-3;                 % gradient clipping

gpus                = 1;                    % gpu on / off
train               = struct('gpus', gpus);

% expdir              = 'training_1';         % experiment name
% if ~exist(expdir, 'dir'), mkdir(expdir); end;
method = 'residual';
Layer = 15; % 2;
overlap = 10; %5;
%% Denoising module: wavresnet or fbpconvnet
printm('load testing data ... \n');
hu2inten = @(x) x./(1000/0.0192);
inten2hu = @(x) x.*(1000/0.0192);

%%
printm 'load transforms ...'
load('Learned_ULTRA/mayo_18s6patSort_block5_iter1000_gamma125_31l0.mat');
mOmega = info.mOmega; clear info;
vLambda = [];
for k = 1:size(mOmega, 3)
    vLambda = cat(1, vLambda, eig(mOmega(:,:,k)' * mOmega(:,:,k)));
end
maxLambda = max(vLambda); clear vLambda;

%% pwls-ultra parameters
ImgSiz =  [ig.nx ig.ny];  % image size
PatSiz =  [8 8];         % patch size
SldDist = [1 1];         % sliding distance

beta_recon = 5e3; % need to tune
gamma_recon = 20; %30; % need to tune

nblock = 1;            % Subset Number
nIter = 5;             % I--Inner Iteration (rlalam)
nOuterIter = 20; %50; %200;     % T--Outer Iteration
CluInt = nIter;            % Clustering Interval
% isCluMap = 0;          % The flag of caculating cluster mapping
% pixmax = inf;          % Set upper bond for pixel values
KapType = 1; % 0: no kappa; 1: with kappa
Ab = Gblock(A, nblock); clear A

mulist = 5e5; %[1e2,5e4];
for imu = 1:length(mulist)
mu = mulist(imu);

netpath = 'trained_model/super_wrn_ultra';
savepath = 'super_wrn_ultra_test/'; 
if ~exist(savepath,'dir') mkdir(savepath); end

%% pre-compute D_R
fprintf('pre-compute D_R\n');
numBlock = size(mOmega, 3);

PP = im2colstep(ones(ImgSiz,'single'), PatSiz, SldDist);
PatNum = size(PP, 2);
%%
printm('load testing data ... \n');
testlist = 100; % [20,50,100,150,200];
% slice = 100;
printm('load testing data ... \n');
caselist = {'L067','L143','L192','L310'}; % 'L067'
layerStart = 0;
for ilist = 1:length(caselist) % 7:10; %1:3
        study = caselist{ilist};
        load(['~/Desktop/MayoData_gen/' study  '/full_3mm_img.mat']);
        load(['~/Desktop/MayoData_gen/' study  '/sim_low_nufft_1e4/xfbp.mat']);
        load(['~/Desktop/MayoData_gen/' study  '/sim_low_nufft_1e4/kappa.mat']);
        load(['~/Desktop/MayoData_gen/' study  '/sim_low_nufft_1e4/denom.mat']);
        load(['~/Desktop/MayoData_gen/' study  '/sim_low_nufft_1e4/sino.mat']);
        load(['~/Desktop/MayoData_gen/' study  '/sim_low_nufft_1e4/wi.mat']);
  
    
    for itestlist = 1:length(testlist)
    test_slice = testlist(itestlist); %100; %2; %#150;
    xref = xfdk(:,:,test_slice);
     kappa_low = kappa(:,:,test_slice);
     denom_low = denom(:,:,test_slice);
     sino_low = sino(:,:,test_slice);
     wi_low = wi(:,:,test_slice);
    if layerStart==0
     xfbp_low = xfbp(:,:,test_slice);
     info_val = [];
    xrla_msk = xfbp_low(ig.mask);
    info_val.RMSE(1) = norm(xrla_msk - xref(ig.mask))/SqrtPixNum;
    info_val.SSIM(1) = ssim(xfbp_low,xref);
    info_val.PSNR(1) = computeRegressedSNR(xrla_msk,xref(ig.mask));
        info_val.cost = [];

    else
        load([savepath study sprintf('s%d_epoch%d_Layer%d_bt%d_gm%d.mat',test_slice,num_epoch,layerStart,beta_recon,gamma_recon)],'info_val');
        xfbp_low = info_val.xultra(:,:,layerStart);
        figure(1); imshow(xfbp_low,[800 1200]);
    end

    xfbp_low = hu2inten(xfbp_low);
    for ilayer = layerStart+1:Layer
        load ([netpath sprintf('/net-Epoch%d-layer-%d.mat',num_epoch,ilayer)]);
        net = vl_simplenn_move(net,'gpu');
        xini= cnn_CT_denoising_forward_process(net,xfbp_low,lv,dflt,patchsize,batchsize,overlap,wgt,gpus);

        xrla = inten2hu(xini); clear xini;
        xrla_msk = xrla(ig.mask);
        info_val.xsup(:,:,ilayer) = xrla;
        fprintf('PWLS-ULTRA begins...\n');
            KapPatch = im2colstep(kappa_low, PatSiz, SldDist);
            KapPatch = mean(KapPatch,1);
                     % construct regularizer R(x)
             R = Reg_OST_Kappa(ig.mask, ImgSiz, PatSiz, SldDist, beta_recon, gamma_recon, KapPatch, mOmega, numBlock, CluInt);
             KapPatch = col2imstep(single(repmat(KapPatch, prod(PatSiz), 1)), ImgSiz, PatSiz, SldDist);
             D_R = 2 * beta_recon * KapPatch(ig.mask) * maxLambda;

             [xrla, info]=pwls_ultra_module(xrla,xref,Ab,sino_low,wi_low,R,nIter,nOuterIter, denom_low,D_R,mu,SqrtPixNum,PatNum,ig);
 

        info_val.xultra(:,:,ilayer) = xrla;
        info_val.RMSE = cat(1,info_val.RMSE,info.RMSE.');
        info_val.SSIM = cat(1,info_val.SSIM,info.SSIM.');
        info_val.PSNR = cat(1,info_val.PSNR,info.PSNR.');
        info_val.cost = cat(2,info_val.cost,info.cost);

        xfbp_low = hu2inten(xrla);
    end
	save([savepath study sprintf('s%d_epoch%d_Layer%d_bt%d_gm%d.mat',test_slice,num_epoch,Layer,beta_recon,gamma_recon)],'info_val');

    %%	
    figure(3);plot(info_val.RMSE,'*-');
    xlabel('number of iterations');
    ylabel('RMSE (HU)'); grid on;
    %export_fig([savepath 'RMSE_' study sprintf('s%d_epoch%d_ULTRA%d_Layer%d_bt%d_gm%d.pdf',test_slice,num_epoch,nOuterIter,Layer,beta_recon,gamma_recon)],'-transparent');
clear info_val;

    end
end
end
