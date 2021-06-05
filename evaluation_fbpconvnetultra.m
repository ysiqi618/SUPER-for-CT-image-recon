%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This is the code for evaluation of SUPER-FCN-ULTRA.
% Pre-trained layer-wise neural networks and pre-learned union of
% sparsifying transforms are required.
%
% modified from Kyong Jin's FBPConvNet testing code
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;clc
restoredefaultpath
reset(gpuDevice)
run('/home/matconvnet-1.0-beta24/matlab/vl_setupnn.m');
run ~/Desktop/irt/setup.m

slice_index = 1;   
for sample = [1,2,3,4,5]
    for patient = {'L192','L143', 'L067', 'L310'}
        cmode='gpu'; % 'cpu'
             
        fprintf('%d th slice of patient %s \n',sample, patient{1})
        addpath(genpath('~/Desktop/toolbox'))
        down = 1; % downsample rate
        sg = sino_geom('fan', 'units', 'mm', 'nb',736, 'na',1152,'orbit',360, 'ds',1.2858,...
             'strip_width','ds','dsd',1085.6,'dso',595,'dfs',0, 'down', down);
        mm2HU = 1000 / 0.0192;
        ig = image_geom('nx',512,'fov',sg.rfov*sqrt(2)); 
        %A = Gtomo2_dscmex(sg, ig, 'nthread', jf('ncore')*2); 
        A = Gtomo_nufft_new(sg, ig); 
        ImgSiz =  [ig.nx ig.ny];  % image size
        PatSiz =  [8 8];         % patch size
        SldDist = [1 1];         % sliding distance
        W = 512;

        beta_recon = 5e3; % need to tune
        gamma_recon = 20; % need to tune
        mu_recon = 5e5;
        Iter = 15;

        nblock = 1;            % Subset Number
        nIter = 5;             % I--Inner Iteration (rlalam)
        nOuterIter = 20;     % T--Outer Iteration
        CluInt = 1;            % Clustering Interval
        KapType = 1; % 0: no kappa; 1: with kappa

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % load pre-learned union of sparsifying transforms (mOmega) here
        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % load normal dose and low dose images here
        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         lab_n = xtrue(:,:,sample);
         lab_d = xfbp_low(:,:,sample);
         lab_n = reshape(lab_n, W, W, 1, []);
         lab_d = reshape(lab_d, W, W, 1, []);
         
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % load kappa, denom, sinogram, and weight matrix here
        %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         kappa = kappa_low(:,:,sample);
         denom = denom_low(:,:,sample);
         sino = sino_low(:,:,sample);
         wi = wi_low(:,:,sample);

        avg_psnr_m=zeros(numel(sample),1);
        avg_psnr_rec=zeros(numel(sample),1);

        SqrtPix = sqrt(numel(ig.mask));
        Ab = Gblock(A, nblock);
        numBlock = size(mOmega, 3);
        vLambda = [];
        for k = 1:size(mOmega, 3)
            vLambda = cat(1, vLambda, eig(mOmega(:,:,k)' * mOmega(:,:,k)));
        end
        maxLambda = max(vLambda); clear vLambda;

        PP = im2colstep(ones(ImgSiz,'single'), PatSiz, SldDist);
        KK = col2imstep(single(PP), ImgSiz, PatSiz, SldDist);

        KapPatch = im2colstep(single(kappa), PatSiz, SldDist); 
        KapPatchh = mean(KapPatch,1);
        Kappa = col2imstep(single(repmat(KapPatchh, prod(PatSiz), 1)), ImgSiz, PatSiz, SldDist);

        switch KapType

            case 0 
                D_R = 2 * beta_recon * KK(:) * maxLambda; 
                R = Reg_OST(ig.mask, ImgSiz, PatSiz, SldDist, beta_recon, gamma_recon, mOmega, numBlock, CluInt);

            case 1  
                Kappa_slice = Kappa;
                D_R = 2 * beta_recon * Kappa_slice(:) * maxLambda;  
                R = Reg_OST_Kappa(ig.mask, ImgSiz, PatSiz, SldDist, beta_recon, gamma_recon, KapPatchh, mOmega, numBlock, CluInt);
        end

        gt=lab_n;
        cost = [];
        for iter=1:Iter
            fprintf('Iter = %d \n',iter);
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %
            % load pre-trained layer-wise neural networks here
            %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            if strcmp(cmode,'gpu')
                net = vl_simplenn_move(net, 'gpu') ;
            else
                net = vl_simplenn_move(net, 'cpu') ;
            end
            if iter == 1
                m=lab_d;
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
            
            xx(:,:,2*iter-1) = temp;
            CCC(1,2*iter-1) = norm(rec(:) - gt(:)) / SqrtPix;
            DDD(1,2*iter-1) = ssim(rec,gt);
            PPP(1,2*iter-1) = computeRegressedSNR(rec,gt);
            fprintf('RMSE = %g, SSIM = %g, PSNR = %g \n', CCC(1,2*iter-1), DDD(1,2*iter-1), PPP(1,2*iter-1));
            
            for iii = 1 : nOuterIter
                fprintf('ULTRA Module %d of %d: \n', iii, nOuterIter);     
                [xrla(:,iii) cost_tmp] = pwls_os_rlalm_l2normReg(temp(:), Ab, reshaper(sino, '2d'),  reshaper(wi, '2d'),...
                R, denom(:), D_R, mu_recon, 'pixmax', inf, 'chat', 1, 'alpha', 1.999, 'rho', [],'niter', nIter);

                [perc, vIdx] = R.nextOuterIter();

                temp = reshape(xrla(:,iii),512,512);

                BBB(1,iii) = norm(temp(:) - gt(:)) / SqrtPix;
                EEE(1,iii) = ssim(temp, gt);
                FFF(1,iii) = computeRegressedSNR(temp,gt);
                fprintf('RMSE = %g, SSIM = %g, PSNR = %g \n', BBB(1,iii), EEE(1,iii), FFF(1,iii));
            end
            cost_ttmp = cost_tmp(1) + mu_recon * sum(col(xrla(:,nOuterIter) - rec(:)).^2);
            cost = [cost cost_ttmp];
            
            CCC(1,2*iter) = BBB(1,nOuterIter);
            DDD(1,2*iter) = EEE(1,nOuterIter);
            PPP(1,2*iter) = FFF(1,nOuterIter);
            m = temp;
            xx(:,:,2*iter) = temp;   

            snr_m=computeRegressedSNR(lab_d,gt);
            snr_rec=computeRegressedSNR(temp,gt);
            figure(1), 
            subplot(131), imshow(lab_d,[800 1200]),axis equal tight, title({'fbp';num2str(snr_m)})
            subplot(132), imshow(temp,[800 1200]),axis equal tight, title({'fbpconvnetultra';num2str(snr_rec)})
            subplot(133), imshow(gt,[800 1200]),axis equal tight, title(['gt ' num2str(iter)])
            pause(0.1)

            avg_psnr_m=snr_m;
            avg_psnr_rec=snr_rec;
        end
        CCC = [RMSE_orig CCC];
        DDD = [SSIM_orig DDD];
        PPP = [snr_m PPP];
        info = struct('xx',xx,'RMSE',CCC,'SSIM',DDD,'snr_m',snr_m,'snr_rec',snr_rec, 'perc',perc,'vIdx',vIdx,'PSNR',PPP,'cost',cost);
        display(['avg SNR (FBP) : ' num2str(mean(avg_psnr_m))])
        display(['avg SNR (FBPconvNet) : ' num2str(mean(avg_psnr_rec))])
        save(sprintf('SUPERULTRA_l2normReg_nufft_500slices_%sSlice%d_MayoST_beta%d_gamma%d_mu%d_nblock%d_nIter%d_nOuterIter%d.mat', ...
             patient{1},slice(sample),beta_recon, gamma_recon, mu_recon, nblock, nIter, nOuterIter),'info');
         
        x_record(:,:,slice_index) = temp;
        x_true(:,:,slice_index) = gt;
        value(1,slice_index) = CCC(end);
        value(2,slice_index) = DDD(end);
        value(3,slice_index) = snr_rec;
        ini_value(1,slice_index) = RMSE_orig;
        ini_value(2,slice_index) = SSIM_orig;
        ini_value(3,slice_index) = snr_m;
        slice_index = slice_index + 1;
        
        clearvars CCC DDD PPP
    end
end

summary = struct('x_record',x_record,'x_true',x_true,'value',value,'ini_value',ini_value);
save('summary_all_samples.mat','summary');
         
