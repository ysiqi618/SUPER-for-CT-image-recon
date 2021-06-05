%%%%%%%%%%%%%%%%%%%%%%%% PWLS-EP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear ; close all; 
addpath(genpath('~/Desktop/toolbox'));
run '~/irt2020/setup.m';
addpath(genpath('~/Desktop/exportfig'));

%% setup target geometry and weight
down = 1; % downsample rate
sg = sino_geom('fan', 'units', 'mm', 'nb',736, 'na',1152,'orbit',360, 'ds',1.2858,...
     'strip_width','ds','dsd',1085.6,'dso',595,'dfs',0, 'down', down);
mm2HU = 1000 / 0.0192;
% ig = image_geom('nx', 512, 'dx', 500/512);
ig = image_geom('nx',512,'fov',sg.rfov*sqrt(2)); % dx=0.69298mm, fov=354.8065mm
% ig.mask = ig.circ>0;

A = Gtomo_nufft_new(sg, ig);
%%
nIter = 100;
nblock = 1; 
l2b = 16;
delta = 2e1; % 10 HU
pot_arg = {'hyper3', delta}; % used in 2D
%%
slice = [20,50,100,150,200];
printm('load testing data ... \n');
caselist = {'L067','L143','L192','L310'};
for ilist = 1:length(caselist) % 7:10; %1:3
        study = caselist{ilist};
        load(['/home/share/MayoData_gen/' study  '/full_3mm_img.mat']);
        load(['/home/share/MayoData_gen/' study  '/sim_low_nufft_1e4/xfbp.mat']);
        load(['/home/share/MayoData_gen/' study  '/sim_low_nufft_1e4/sino.mat']);
        load(['/home/share/MayoData_gen/' study  '/sim_low_nufft_1e4/wi.mat']);
        load(['/home/share/MayoData_gen/' study  '/sim_low_nufft_1e4/denom.mat']);
        load(['/home/share/MayoData_gen/' study  '/sim_low_nufft_1e4/kappa.mat']);


        for itest = 1:length(slice)
            sample = slice(itest);
           
            xtrue = xfdk(:,:,sample);
             xfbp_low = xfbp(:,:,sample);
            
             kappa_low = kappa(:,:,sample);
             denom_low = denom(:,:,sample);
             sino_low = sino(:,:,sample);
             wi_low = wi(:,:,sample);

            %% setup edge-preserving regularizer
           R = Reg1(sqrt(kappa_low), 'beta', 2^l2b, 'pot_arg', pot_arg, 'nthread', jf('ncore'));

            fprintf('iteration begins...\n'); 
            [xrlalm_msk , info] = pwls_ep_os_rlalm_l2normReg(xfbp_low(:), A, sino_low, R,0,'wi', wi_low, ...
                        'pixmax', inf, 'isave', 'last',  'niter', nIter, 'nblock', nblock, ...
                        'chat', 0, 'denom',denom_low, 'xtrue', xtrue, 'mask', ig.mask);
            
            xrlalm = ig.embed(xrlalm_msk);
             figure name 'xrlalm'
             imshow(cat(2, xrlalm(:,:,end), xfbp_low), [800 1200]);colorbar;
           export_fig(['pwls-ep-nufft/1e4/fig_' study sprintf('_s%d_l2b%d_del%d_iter%d.pdf',sample,l2b,delta,nIter)],'-transparent');
            
            info.xx = xrlalm;
            % save(sprintf('PWLS-EP/1e4/L067_Slice%d_l2b%d_del%d_iter%d.mat',sample,l2b,delta,nIter),'info');
            % save(sprintf('PWLS-EP/1e4/L192_Slice%d_l2b%d_del%d_iter%d.mat',sample,l2b,delta,nIter),'info');
            save(['pwls-ep-nufft/1e4/' study sprintf('_s%d_l2b%d_del%d_iter%d.mat',sample,l2b,delta,nIter)],'info');
            
            end
end
