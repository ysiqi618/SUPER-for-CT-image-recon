%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Ye Siqi, UM-SJTU Joint Institute
clear ; close all;
run '~/irt2020/setup.m';
addpath(genpath('~/exportfig'));

addpath(genpath('~/exportfig'));
addpath(genpath('~/Desktop/toolbox'));
addpath(genpath('/home/share/MayoData_gen/'));
%% setup target geometry and weight
down = 1; % downsample rate
sg = sino_geom('fan', 'units', 'mm', 'nb',736, 'na',1152,'orbit',360, 'ds',1.2858,...
     'strip_width','ds','dsd',1085.6,'dso',595,'dfs',0, 'down', down);
mm2HU = 1000 / 0.0192;
% ig = image_geom('nx', 512, 'dx', 500/512);
ig = image_geom('nx',512,'fov',sg.rfov*sqrt(2)); % dx=0.69298mm, fov=354.8065mm
% ig.mask = ig.circ>0;
% A = Gtomo2_dscmex(sg, ig);
A = Gtomo_nufft_new(sg, ig);

%% load external parameter
dose = '1e4'; % I0, photon intensity
snappath = ['pwls-ultra-nufft/' dose '/snapshot/'];
if ~exist(snappath,'dir') mkdir(snappath); end
% load('Learned_ULTRA/mayo_18s6pat_block5_iter1000_gamma125_31l0.mat');
load('Learned_ULTRA/mayo_18s6patSort_block5_iter1000_gamma125_31l0.mat');
mOmega = info.mOmega; clear info;
%% setup edge-preserving regularizer
ImgSiz =  [ig.nx ig.ny];  % image size
PatSiz =  [8 8];         % patch size
SldDist = [1 1];         % sliding distance

nblock = 1;            % Subset Number
nIter = 5;             % I--Inner Iteration
nOuterIter = 1000;     % T--Outer Iteration
CluInt = 5;            % Clustering Interval
isCluMap = 0;          % The flag of caculating cluster mapping
pixmax = inf;          % Set upper bond for pixel values
% Ab = Gblock(A, nblock); clear A


% pre-compute D_R
numBlock = size(mOmega, 3);
vLambda = [];
for k = 1:size(mOmega, 3)
    vLambda = cat(1, vLambda, eig(mOmega(:,:,k)' * mOmega(:,:,k)));
end
maxLambda = max(vLambda); clear vLambda;

PP = im2colstep(ones(ImgSiz,'single'), PatSiz, SldDist);
PatNum = size(PP, 2);
KK = col2imstep(single(PP), ImgSiz, PatSiz, SldDist);
%%
slice = [20,50,100,150,200]; % 20
printm('load testing data ... \n');
caselist = {'L067','L143','L192','L310'};
for ilist = 1:length(caselist) % 7:10; %1:3
        study = caselist{ilist};
        fprintf(['start ' study '...\n']);
        load(['dataset/test_nufft_' dose '/' study '.mat']);
 for itest = 1:size(xfbp_low,3)
        sample = slice(itest);
        xref = xtrue(:,:,itest);
        xrlalm = xfbp_low(:,:,itest);
        kappa = single(kappa_low(:,:,itest));
        denom = denom_low(:,:,itest);
        sino = sino_low(:,:,itest);
        wi = wi_low(:,:,itest);
        % load([dir '/kappa_low_low.mat']);
         KapPatch = im2colstep(kappa, PatSiz, SldDist); clear kappa;
         KapPatch = mean(KapPatch,1);
         %         KapPatch = repmat(KapPatch, prod(PatSiz), 1);
         Kappa = col2imstep(single(repmat(KapPatch, prod(PatSiz), 1)), ImgSiz, PatSiz, SldDist);
       
         KapType = 1;
        switch KapType
            case 0  % no patch-based weighting
                beta = 5e4;
                gamma = 80;

                D_R = 2 * beta * KK(ig.mask) * maxLambda; clear PP KK
                % D_R = 2 * beta * prod(PatSiz)/ prod(SldDist) * maxLambda;
                R = Reg_OST(ig.mask, ImgSiz, PatSiz, SldDist, beta, gamma, mOmega, numBlock, CluInt);

            case 1  % patch-based weighting \tau * { \|~~\|_2 + \|~~\|_0 }
                beta = 5e3; %5e3; %5e3; %5e3; %5e3; %1e4;
                gamma = 30;

                D_R = 2 * beta * Kappa(ig.mask) * maxLambda;  clear Kappa;
                % construct regularizer R(x)
                R = Reg_OST_Kappa(ig.mask, ImgSiz, PatSiz, SldDist, beta, gamma, KapPatch, mOmega, numBlock, CluInt);

        end
        fprintf('KapType= %g, beta = %.1e, gamma = %g: \n\n', KapType, beta, gamma);


        info = struct('ImgSiz',ImgSiz,'SldDist',SldDist,'beta',beta,'gamma',gamma,...
            'nblock',nblock,'nIter',nIter,'CluInt',CluInt,'pixmax',pixmax,'transform',mOmega,...
            'xrla',[],'vIdx',[],'ClusterMap',[], 'RMSE',[],'SSIM',[],'PSNR',[],'relE',[],'perc',[],'idx_change_perc',[],'cost',[]);

        xini = xrlalm .* ig.mask;    %initial EP image
        xrla_msk = xrlalm(ig.mask);
        % xini = xfbp .* ig.mask;     %initial FBP image
        % xrla_msk = xfbp(ig.mask);   clear xfbp
        xrla = xini;

    %% Recon
    SqrtPixNum = sqrt(sum(ig.mask(:)>0)); % sqrt(pixel numbers in the mask)
    stop_diff_tol = 1e-3; % HU
    idx_old = ones([1,PatNum],'single');

    for ii=1:nOuterIter
        %     figure(iterate_fig); drawnow;
        xold = xrla_msk;
        AAA(1,ii) = norm(xrla_msk - xref(ig.mask)) / SqrtPixNum;
        fprintf('RMSE = %g, ', AAA(1,ii));
        info.RMSE = AAA(1,:);
        AAA(2,ii)= ssim(xrla, xref);
        fprintf('SSIM = %g\n', AAA(2,ii));
        info.SSIM = AAA(2,:);
        snr_rec=computeRegressedSNR(xrla_msk,xref(ig.mask));
        info.PSNR(ii) = snr_rec;

        fprintf('Iteration = %d:\n', ii);
        [xrla_msk, cost] = pwls_os_rlalm_l2normReg(xrla_msk, A, reshaper(sino, '2d'),  reshaper(wi, '2d'),...
            R, denom, D_R, 0,'pixmax', pixmax, 'chat', 0, 'alpha', 1.999, 'rho', [],'niter', nIter);

        [info.perc(:,ii),info.vIdx] = R.nextOuterIter();
        fprintf('perc = %g\n', info.perc(:,ii));

        info.idx_change_perc(:,ii) = nnz(idx_old - info.vIdx)/PatNum;
        fprintf('Idx Change Perc = %g\n', info.idx_change_perc(:,ii));
        idx_old = info.vIdx;

        %     info.cost(:,ii) = cost;
        info.relE(:,ii) =  norm(xrla_msk - xold) / SqrtPixNum;
        fprintf('relE = %g\n', info.relE(:,ii));
        if info.relE(:,ii) < stop_diff_tol
            break
        end
      xrla= ig.embed(xrla_msk);

        % figure(120), imshow(xrla, [800 1200]); drawnow;
        %     info.ClusterMap = ClusterMap(ImgSiz, PatSiz, SldDist, info.vIdx, PatNum, numBlock);
        %     figure(iterate_fig); drawnow;
        %     subplot(2,3,1),imshow((info.ClusterMap == 1) .* info.xrla, [800,1200]);
        %     subplot(2,3,2);imshow((info.ClusterMap == 2) .* info.xrla, [800,1200]);
        %     subplot(2,3,3);imshow((info.ClusterMap == 3) .* info.xrla, [800,1200]);
        %     subplot(2,3,4);imshow((info.ClusterMap == 4) .* info.xrla, [800,1200]);
        %     subplot(2,3,5);imshow((info.ClusterMap == 5) .* info.xrla, [800,1200]);
        if mod(ii,10)==0
            info.xrla = xrla;
            figure(120);imshow(xrla,[800 1200]);drawnow;
            export_fig(['pwls-ultra-nufft/' dose '/snapshot/' study sprintf('_s%d_mayo18s6pSort_beta%.1e_gm%d_cInt%d_inIter%d.pdf',sample,beta,gamma,CluInt,ii)],'-transparent');
    %  	 save(sprintf('./PWLS-ULTRA/1e4/L067Slice%d_mayo18s6pSort_beta%.1e_gm%d_cInt%d_inIter%d_info.mat',sample,beta,gamma,CluInt,nIter), 'info','-v7.3');
            save(['pwls-ultra-nufft/' dose '/' study sprintf('_s%d_mayo18s6pSort_beta%.1e_gm%d_cInt%d_inIter%d_info.mat',sample,beta,gamma,CluInt,nIter)], 'info','-v7.3');

        end
    
    end
        figure(120);imshow(xrla,[800 1200]);drawnow;
        export_fig(['pwls-ultra-nufft/' dose '/' study sprintf('_s%d_mayo18s6pSort_beta%.1e_gm%d_cInt%d_inIter%d.pdf',sample,beta,gamma,CluInt,nIter)],'-transparent');

     info.xrla = xrla;
% 	 save(sprintf('./PWLS-ULTRA/1e4/L067Slice%d_mayo18s6pSort_beta%.1e_gm%d_cInt%d_inIter%d_info.mat',sample,beta,gamma,CluInt,nIter), 'info','-v7.3');
	 save(['pwls-ultra-nufft/' dose '/' study sprintf('_s%d_mayo18s6pSort_beta%.1e_gm%d_cInt%d_inIter%d_info.mat',sample,beta,gamma,CluInt,nIter)], 'info','-v7.3');

 end
end
