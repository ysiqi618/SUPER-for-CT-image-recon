% evaluation - FBPConvNet
% modified from MatconvNet (ver.23)
% 22 June 2017
% modified based on Kyong Jin's fbpconvnet evaluation code

clear; clc
restoredefaultpath
reset(gpuDevice(1))
run('/home/matconvnet-1.0-beta24/matlab/vl_setupnn.m');
datafolder = '../data/';
slice_index = 1;
W = 512;
SqrtPix = 512;


for sample = [20,50,100,150,200]
    for patient = {'L192','L143', 'L067', 'L310'}
        
        fprintf('%d th slice of patient %s \n',sample, patient{1})
        ttemp = strcat('../trained_model/fbpconvnet_standalone/net-epoch-101.mat');
        load(ttemp);         
        %%%%%%%%%%%%%%%% load test data here %%%%%%%%%%%%%%%%%%%
    	%datafolder = "/home/share/MayoData_gen";  % this will be changed by users

        load([datafolder patient{1} '/full_3mm_img.mat']);
        load([datafolder patient{1} '/sim_low_nufft_1e4/xfbp.mat']);
        load([datafolder patient{1} '/sim_low_nufft_1e4/sino.mat']);
        load([datafolder patient{1} '/sim_low_nufft_1e4/wi.mat']);
        load([datafolder patient{1} '/sim_low_nufft_1e4/denom.mat']);
        load([datafolder patient{1} '/sim_low_nufft_1e4/kappa.mat']);    
                
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
        lab_n = reshape(xfdk, W, W, 1, []); % true
        lab_d = reshape(xfbp, W, W, 1, []); % fbp

        cmode='gpu'; % 'cpu'
        if strcmp(cmode,'gpu')
            net = vl_simplenn_move(net, 'gpu') ;
        else
            net = vl_simplenn_move(net, 'cpu') ;
        end

        avg_psnr_m=zeros(1,1);
        avg_psnr_rec=zeros(1,1);

        
        gt=lab_n(:,:,1,sample);
        m=lab_d(:,:,1,sample);
        RMSE(1,1) = norm(m(:)-gt(:)) / SqrtPix;
        SSIM(1,1) = ssim(m, gt);
        if strcmp(cmode,'gpu')
            res=vl_simplenn_fbpconvnet_eval(net,gpuArray((single(m))));
            rec=gather(res(end-1).x)+m;
        else
            res=vl_simplenn_fbpconvnet_eval(net,((single(m))));
            rec=(res(end-1).x)+m;
        end

        RMSE(1,2) = norm(rec(:)-gt(:)) / SqrtPix;
        SSIM(1,2) = ssim(rec, gt);
        snr_m=computeRegressedSNR(m,gt);
        snr_rec=computeRegressedSNR(rec,gt);
        figure(1), 
        subplot(131), imshow(m,[800 1200]),axis equal tight, title({'fbp';num2str(snr_m)})
        subplot(132), imshow(rec,[800 1200]),axis equal tight, title({'fbpconvnet';num2str(snr_rec)})
        subplot(133), imshow(gt,[800 1200]),axis equal tight
        pause(0.1)

        avg_psnr_m=snr_m;
        avg_psnr_rec=snr_rec;
     
        info = struct('rec',rec,'m',m,'RMSE',RMSE,'SSIM',SSIM,'snr_m',snr_m,'snr_rec',snr_rec,'gt',gt);
        display(['avg SNR (FBP) : ' num2str(mean(avg_psnr_m))])
        display(['avg SNR (FBPconvNet) : ' num2str(mean(avg_psnr_rec))])
        
       %save(sprintf('FBPConvNet_%sSlice%d_dose1e4.mat', ...
       %    patient{1},sample),'info');
        
        x_record(:,:,slice_index) = rec;
        x_true(:,:,slice_index) = gt;
        value(1,slice_index) = RMSE(1,end);
        value(2,slice_index) = SSIM(1,end);
        value(3,slice_index) = snr_rec;
        slice_index = slice_index + 1;
    end
end
%summary = struct('x_record',x_record,'x_true',x_true,'value',value);

