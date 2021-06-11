%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This is the test code for successive FCN.
% Pre-trained layer-wise nrural networks are required.
%
% modified from Kyong Jin's FBPConvNet test code
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;clc
restoredefaultpath
reset(gpuDevice)

run /home/matconvnet-1.0-beta24/matlab/vl_setupnn.m
run ~/Desktop/irt/setup.m
datafolder = '../data/';
addpath(genpath('/home/matconvnet-1.0-beta24/matlab/src'))
slice_index = 1;
for sample = [20,50,100,150,200]
    for patient = {'L192','L143', 'L067', 'L310'}
        
        cmode='gpu'; % 'cpu'
        fprintf('%d th slice of patient %s \n',sample, patient{1})
        down = 1; % downsample rate
        sg = sino_geom('ge1', 'units', 'mm', 'strip_width', 'd', 'down', down);
        mm2HU = 1000 / 0.0192;
        ig = image_geom('nx', 512, 'dx', 500/512);
        A = Gtomo2_dscmex(sg, ig);  
        ImgSiz =  [ig.nx ig.ny];  % image size
        PatSiz =  [8 8];         % patch size
        SldDist = [1 1];         % sliding distance
        W = 512;

        Iter = 15;
        
        % --------------------------------------------------------
        %
        % load normal dose and low dose images here
        %
        load([datafolder patient{1} '/full_3mm_img.mat']);
        load([datafolder patient{1} '/sim_low_nufft_1e4/xfbp.mat']);
        % ---------------------------------------------------------
        lab_n = xfdk;
        lab_d = xfbp;
        lab_n = reshape(lab_n, W, W, 1, []);
        lab_d = reshape(lab_d, W, W, 1, []);

        avg_psnr_m=zeros(numel(sample),1);
        avg_psnr_rec=zeros(numel(sample),1);

        SqrtPix = sqrt(numel(ig.mask));

        gt=lab_n(:,:,1,sample);
        for iter=1:Iter
            fprintf('Iter = %d \n',iter);
            
            % ----------------------------------------------
            %
            % load pre-trained layer-wise neural networks here
            %
            ttemp = strcat('../trained_model/fbpconvnet_sequential/net-epoch-', num2str(iter) ,'.mat');
            load(ttemp); 
            % -----------------------------------------------

            if strcmp(cmode,'gpu')
                net = vl_simplenn_move(net, 'gpu') ;
            else
                net = vl_simplenn_move(net, 'cpu') ;
            end
            if iter == 1
                m=lab_d(:,:,1,sample);
                RMSE_orig = norm(m(:) - gt(:)) / SqrtPix;
                SSIM_orig = ssim(m,gt);
                fprintf('RMSE = %g, SSIM = %g \n', RMSE_orig, SSIM_orig);
            end
            if strcmp(cmode,'gpu')
                res=vl_simplenn_fbpconvnet_eval(net,gpuArray((single(m))));
                rec=gather(res(end-1).x)+m;
            else
                res=vl_simplenn_fbpconvnet_eval(net,((single(m))));
                rec=(res(end-1).x)+m;
            end
            temp = rec;
            xx(:,:,iter) = temp;
            CCC(1,iter) = norm(rec(:) - gt(:)) / SqrtPix;
            DDD(1,iter) = ssim(rec,gt);
            EEE(1,iter) = computeRegressedSNR(rec,gt);
            fprintf('RMSE = %g, SSIM = %g \n', CCC(1,iter), DDD(1,iter));

            m = temp;
            xx(:,:,iter) = temp; 

            snr_m=computeRegressedSNR(lab_d(:,:,1,sample),gt);
            snr_rec=computeRegressedSNR(temp,gt);
            figure(1), 
            subplot(131), imshow(lab_d(:,:,1,sample),[800 1200]),axis equal tight, title({'fbp';num2str(snr_m)})
            subplot(132), imshow(temp,[800 1200]),axis equal tight, title({'fbpconvnet';num2str(snr_rec)})
            subplot(133), imshow(gt,[800 1200]),axis equal tight, title(['gt ' num2str(iter)])
            pause(0.1)

            avg_psnr_m=snr_m;
            avg_psnr_rec=snr_rec;
        end

        info = struct('xx',xx,'RMSE',CCC,'SSIM',DDD,'PSNR',EEE,'snr_m',snr_m,'snr_rec',snr_rec);
        display(['avg SNR (FBP) : ' num2str(mean(avg_psnr_m))])
        display(['avg SNR (FBPconvNet) : ' num2str(mean(avg_psnr_rec))])
        %save(sprintf('PUREFBPConvNet_4epoch_15layer_%sSlice%d_Dose1e4.mat', patient{1},sample),'info');
        
        x_record(:,:,slice_index) = temp;
        x_true(:,:,slice_index) = gt;
        value(1,slice_index) = CCC(end);
        value(2,slice_index) = DDD(end);
        value(3,slice_index) = snr_rec;
        ini_value(1,slice_index) = RMSE_orig;
        ini_value(2,slice_index) = SSIM_orig;
        ini_value(3,slice_index) = snr_m;
        slice_index = slice_index + 1;
    end
end

summary = struct('x_record',x_record,'x_true',x_true,'value',value,'ini_value',ini_value);
%save('summary_all_samples.mat','summary');
