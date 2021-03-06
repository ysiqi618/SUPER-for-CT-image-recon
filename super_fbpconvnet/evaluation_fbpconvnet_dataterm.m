%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This is the code for evaluation of FBPConvNet+DataTerm Only.
% Pre-trained layer-wise neural networks are required.
% modified from evaluation code of SUPER-FCN-EP.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;clc
restoredefaultpath
reset(gpuDevice)

run('~/Desktop/irt/setup.m');
run('/home/matconvnet-1.0-beta24/matlab/vl_setupnn.m');

slice_index = 1;
for sample = [20,50,100,150,200]
    for patient = {'L192','L143', 'L067', 'L310'}
        cmode='gpu'; % 'cpu'
        
        fprintf('%d th slice of patient %s \n',sample, patient{1})
        addpath(genpath('../toolbox'))
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

        beta_recon = 0; % need to tune
        delta_recon = 0; % need to tune
        mu_recon = 0;
        Iter = 15;

        nblock = 1;            % Subset Number
        nOuterIter = 5;     % T--Outer Iteration

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % load paired normal dose and low dose images here
        %
        load(['/home/share/MayoData_gen/' patient{1} '/full_3mm_img.mat']);
        load(['/home/share/MayoData_gen/' patient{1} '/sim_low_nufft_1e4/xfbp.mat']);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         lab_n = xfdk;
         lab_d = xfbp;
         lab_n = reshape(lab_n, W, W, 1, []);
         lab_d = reshape(lab_d, W, W, 1, []);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % load kappa, denom, sinogram, and weight matrix here
        %
        load(['/home/share/MayoData_gen/' patient{1} '/sim_low_nufft_1e4/sino.mat']);
        load(['/home/share/MayoData_gen/' patient{1} '/sim_low_nufft_1e4/wi.mat']);
        load(['/home/share/MayoData_gen/' patient{1} '/sim_low_nufft_1e4/denom.mat']);
        load(['/home/share/MayoData_gen/' patient{1} '/sim_low_nufft_1e4/kappa.mat']);  
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
         kappa = kappa(:,:,sample);
         denom = denom(:,:,sample);
         sino = sino(:,:,sample);
         wi = wi(:,:,sample);


        avg_psnr_m=zeros(numel(sample),1);
        avg_psnr_rec=zeros(numel(sample),1);

        SqrtPix = sqrt(numel(ig.mask));
        Ab = Gblock(A, nblock);

        pot_arg = {'hyper3', delta_recon}; % used in 2D
        R = Reg1(sqrt(kappa), 'beta', 2^beta_recon, 'pot_arg', pot_arg, 'nthread', jf('ncore'));

        gt=lab_n(:,:,1,sample);
        for iter=1:Iter
            fprintf('Iter = %d \n',iter);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %
        % load pre-trained layer-wise neural networks here
        %
            ttemp = strcat('../trained_model/super_fcn_dataterm/net-epoch-', num2str(iter) ,'.mat');
            load(ttemp);  
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

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
           xx(:,:,2*iter-1) = temp;
           CCC(1,2*iter-1) = norm(rec(:) - gt(:)) / SqrtPix;
           DDD(1,2*iter-1) = ssim(rec,gt);
           PPP(1,2*iter-1) = computeRegressedSNR(rec,gt);
           fprintf('RMSE = %g, SSIM = %g, PSNR = %g \n', CCC(1,2*iter-1), DDD(1,2*iter-1), PPP(1,2*iter-1));


            fprintf('MBIR iteration begins...\n'); 
            [xrla , info] = pwls_ep_os_rlalm_noreg(temp(:), A, sino,  [], 'wi', wi, ...
                'pixmax', inf, 'isave', 'last',  'niter', nOuterIter, 'nblock', nblock, ...
                'chat', 0, 'denom',denom, 'xtrue', gt, 'mask', ig.mask);

            temp = reshape(xrla,512,512);

            BBB = norm(temp(:) - gt(:)) / SqrtPix;
            EEE = ssim(temp, gt);
            FFF = computeRegressedSNR(temp,gt);
            fprintf('RMSE = %g, SSIM = %g, PSNR = %g \n', BBB, EEE, FFF);

            CCC(1,2*iter) = BBB;
            DDD(1,2*iter) = EEE;
            PPP(1,2*iter) = FFF;
            m = temp;
            xx(:,:,2*iter) = temp;   


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
        CCC = [RMSE_orig CCC];
        DDD = [SSIM_orig DDD];
        PPP = [snr_m PPP];
        info = struct('xx',xx,'RMSE',CCC,'SSIM',DDD,'snr_m',snr_m,'snr_rec',snr_rec,'PSNR',PPP);
        display(['avg SNR (FBP) : ' num2str(mean(avg_psnr_m))])
        display(['avg SNR (FBPconvNet) : ' num2str(mean(avg_psnr_rec))])
        %save(sprintf('SUPEREP_fcn_dataterm_500slices_%sSlice%d_dose1e4_l2b%d_delta%d_testmu%d_nblock%d_nOuterIter%d.mat', ...
        %    patient{1},sample,beta_recon, delta_recon, mu_recon, nblock, nOuterIter),'info');
        
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
%save('summary_all_samples.mat','summary');

