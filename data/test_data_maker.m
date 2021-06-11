%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Siqi Ye, UM-SJTU Joint Institute
clear;
close all;
run '~/irt2020/setup.m';
%% generate noisy sinogram and statistical weighting

caselist = {'L067'}; %{'L067','L143','L192','L310'}; % four test patient
slice = 20; %[20,50,100,150,200];


I0_low = 1e4; 
foldername = '/sim_low_nufft_1e4/'; 


down = 1; % downsample rate
sg = sino_geom('fan', 'units', 'mm', 'nb',736, 'na',1152,'orbit',360, 'ds',1.2858,...
     'strip_width','ds','dsd',1085.6,'dso',595,'dfs',0, 'down', down); 
mm2HU = 1000 / 0.0192;
ig = image_geom('nx',512,'fov',sg.rfov*sqrt(2)); % dx=0.69298mm, fov=354.8065mm
A = Gtomo_nufft_new(sg, ig); % faster forward-backward operator with nufft


tmp = fbp2(sg, ig);

for ilist = 1:length(caselist)
        study = caselist{ilist};
        savedir = ['data/' study foldername];
        if ~exist(savedir,'dir');mkdir(savedir);end
 %   
 
	load(['data/' study '/full_3mm_img.mat']);
	for i = 1:length(slice)
		fprintf([study ': slice #%d\n'],slice(i));
        
        xfbp_full = reshape(xfdk(:,:,slice(i)),[512 512]);
		sino_full = A * xfbp_full; clear xfbp_full;
	% figure;imshow(xfbp_full,[800 1200]);	
		% fprintf('adding noise...\n');
		yi = poisson(I0_low * exp(-sino_full ./mm2HU), 0, 'factor', 0.5);
		var = 5.^2;
		ye = sqrt(var).* randn(size(yi)); % Gaussian white noise ~ N(0,std^2)
		k = 1;
		zi = k * yi + ye; 
		error = 1/1e1 ;
		zi = max(zi, error);   
		sino(:,:,slice(i)) = -log(zi ./(k*I0_low)) * mm2HU; 
		
		wi(:,:,slice(i)) = (zi.^2)./(k*zi + var);  
		
		%% setup target geometry and fbp
		fprintf('fbp...\n');
		xfbp_tmp = fbp2(sino(:,:,slice(i)), tmp, 'window', 'hanning,0.8'); %0.8 for 1e5 and 1e4 
		xfbp(:,:,slice(i)) = max(xfbp_tmp, 0);
% 		 figure name 'xfbp'
% 		 imshow(xfbp(:,:,i), [800 1200]);
% 	pause;
% 	break;
		%% setup kappa
	 	 fprintf('calculating kappa...\n');
	 	 kappa(:,:,slice(i)) = sqrt( div0(A' * wi(:,:,slice(i)), A' * ones(size(sino_full))) );
	 	 
	 	 %% setup diag{A'WA1}
	 	 fprintf('Pre-calculating denominator D_A...');
	 	 denom(:,:,slice(i)) = A' * col(reshape(sum(A'), size(sino_full)) .* wi(:,:,slice(i))); 
	end
	
	   save([savedir '/sino.mat'],'sino','-v7.3');
	   save([savedir '/xfbp.mat'],'xfbp','-v7.3');
	   save([savedir '/wi.mat'],'wi','-v7.3');
	   save([savedir '/kappa.mat'],'kappa','-v7.3');
	   save([savedir '/denom.mat'],'denom','-v7.3');
	   clear sino xfbp wi kappa denom;
end
