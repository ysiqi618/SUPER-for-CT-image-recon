
clear all;
close all;

%% Path setting
run 'irt/setup.m'; % irt path
run('~/matconvnet-1.0-beta24/matlab/vl_setupnn.m'); % MatConvNet path
addpath(genpath('~/SUPER-for-CT-recon/')); % project path
%  addpath(genpath('/home/share/SUPER/github-code/toolbox/'));
%  addpath(genpath('/home/share/SUPER/github-code/trained_model'));
% 
datafolder = 'data/';
%% geometry setting
down = 1; % downsample rate
sg = sino_geom('fan', 'units', 'mm', 'nb',736, 'na',1152,'orbit',360, 'ds',1.2858,...
     'strip_width','ds','dsd',1085.6,'dso',595,'dfs',0, 'down', down);

ig = image_geom('nx',512,'fov',sg.rfov*sqrt(2)); % dx=0.69298mm, fov=354.8065mmi
SqrtPixNum = sqrt(sum(ig.mask(:)>0));
% A = Gtomo2_dscmex(sg, ig);
A = Gtomo_nufft_new(sg, ig); % faster forward-backward operator with nufft

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
overlap = 10; %5;
%%
mulist = 5e4; %[5e3,1e4,5e4,1e5,1e6,1e7];
for imu = 1:length(mulist)
mu = mulist(imu);
Layer = 15;
nIter = 20;
nblock = 1;
l2b = 16;
delta = 20;
pot_arg = {'hyper3', delta}; % used in 2D
hu2inten = @(x) x./(1000/0.0192);
inten2hu = @(x) x.*(1000/0.0192);
%netpath = sprintf('wavEP/zup_nufft_mu%0.e_l2b16_lr-5_imgs520_PatSz256_Overlap10_BatSz1_WgtDecay1e-02_GradMax1e-03/',mu);
netpath = 'trained_model/super_wrn_ep/';
savepath = 'super_wrn_ep_test/';
if ~exist(savepath,'dir') mkdir(savepath); end
%% Denoising module: wavresnet or fbpconvnet
printm('load testing data ... \n');
slice = [20,50,100,150,200];

printm('load testing data ... \n');
caselist = {'L067','L143','L192','L310'};
for ilist = 1:length(caselist) % 7:10; %1:3
         study = caselist{ilist};
        load([datafolder study  '/full_3mm_img.mat']);
        load([datafolder study '/sim_low_nufft_1e4/xfbp.mat']);
        load([datafolder study '/sim_low_nufft_1e4/sino.mat']);
        load([datafolder study '/sim_low_nufft_1e4/wi.mat']);
        load([datafolder study '/sim_low_nufft_1e4/denom.mat']);
        load([datafolder study '/sim_low_nufft_1e4/kappa.mat']);

 for itest = 1:length(slice)
            test_slice = slice(itest);
            xfbp_low = xfbp(:,:,test_slice);
            xref = xfdk(:,:,test_slice);
             kappa_low = kappa(:,:,test_slice);
             denom_low = denom(:,:,test_slice);
             sino_low = sino(:,:,test_slice);
             wi_low = wi(:,:,test_slice);
R = Reg1(sqrt(kappa_low), 'beta', 2^l2b, 'pot_arg', pot_arg, 'nthread', jf('ncore'));

xrla_msk = xfbp_low(ig.mask);
info_val.RMSE(1) = norm(xrla_msk - xref(ig.mask))/SqrtPixNum;
info_val.SSIM(1) = ssim(xfbp_low,xref);
info_val.PSNR(1) = computeRegressedSNR(xrla_msk,xref(ig.mask));
info_val.cost = [];
 xfbp_low = hu2inten(xfbp_low);
for ilayer = 1:Layer
	load ([netpath sprintf('/net-Epoch%d-layer-%d.mat',num_epoch,ilayer)]);

	net = vl_simplenn_move(net,'gpu');
	xini= cnn_CT_denoising_forward_process(net,xfbp_low,lv,dflt,patchsize,batchsize,overlap,wgt,gpus);
	
	xrla = inten2hu(xini); clear xini;
	xrla_msk = xrla(ig.mask);
	info_val.xx(:,:,2*ilayer-1) = xrla;
	figure(50);
	imshow(xrla,[800 1200]);
    %export_fig([savepath '/wrs_1e4_' study sprintf('_s%d_epoch%d_ilayer%d_EP%d.pdf',test_slice,num_epoch,ilayer,nIter)],'-transparent');

	fprintf('PWLS-EP begins...\n');
    [xrlalm_msk , info] = pwls_ep_os_rlalm_l2normReg(xrla, A, sino_low, R,mu,'wi', wi_low,'pixmax', inf, 'isave', 'last',  'niter', nIter, 'nblock', nblock,'chat', 1, 'denom',denom_low, 'xtrue', xref, 'mask', ig.mask);
	xrla = ig.embed(xrlalm_msk);
	 figure(51)
	 imshow(xrla, [800 1200]);
    %export_fig([savepath '/ep_1e4_' study sprintf('_s%d_epoch%d_ilayer%d_EP%d.pdf',test_slice,num_epoch,ilayer,nIter)],'-transparent');

    info_val.xx(:,:,2*ilayer) = xrla;
	info_val.RMSE = cat(1,info_val.RMSE,info.RMSE.');
	info_val.SSIM = cat(1,info_val.SSIM,info.SSIM.');
	info_val.PSNR = cat(1,info_val.PSNR,info.PSNR.');
    info_val.cost = cat(1,info_val.cost,info.cost.');

	xfbp_low = hu2inten(xrla);
end
    save([savepath study sprintf('_s%d_epoch%d_Layer%d.mat',test_slice,num_epoch,Layer)],'info_val');

clear info_val;
 end
end
end
