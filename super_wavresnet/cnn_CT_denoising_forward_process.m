function recon = cnn_CT_denoising_forward_process(net,noisy,lv,dflt,patchsize,batchsize,overlap,wgt,gpus)


ind         = 1 : size(noisy,3);
noisyCoeffs = single(cnn_wavelet_decon(double(noisy),lv,dflt));
noisyCoeffs = noisyCoeffs .*wgt;

% [ny, nx]    = size(noisy);
ny  = size(noisy,1);
nx = size(noisy,2);
wgtMap      = zeros(ny, nx, 'single');
reconCoeffs = zeros(size(noisyCoeffs),'single');

% if gpus > 0
%     reconCoeffs = gpuArray(reconCoeffs);
% end

for y = 1:(patchsize-overlap):ny-1
    yy = min(y, ny-patchsize+1);
    
    for x = 1:(patchsize-overlap):nx-1
        xx = min(x,nx-patchsize+1);
        
        wgtMap(yy:(yy+patchsize-1),xx:(xx+patchsize-1),:) = ...
            wgtMap(yy:(yy+patchsize-1),xx:(xx+patchsize-1),:) +1;
        
        for tt = 1:ceil(length(ind)/batchsize)
            ind_s = batchsize*(tt-1)+1;
            ind_e = min(length(ind),ind_s+batchsize-1);
       
%             fprintf('y=%d, x=%d, tt=%d, yy=%d, xx=%d, xx_thred=%d, patchsize=%d\n',y,x,tt,yy,xx,nx-patchsize+1,patchsize);
            noisyCoeffsSub = noisyCoeffs(yy:(yy+patchsize-1),xx:(xx+patchsize-1),:,ind_s:ind_e);
            if gpus > 0
                noisyCoeffsSub = gpuArray(noisyCoeffsSub);
                res = vl_simplenn_modified(net,single(noisyCoeffsSub),[],[],...
                'mode','test',...
                'conserveMemory', 1, ...
                'cudnn', 1);
                reconCoeffsSub = gather(res(end-1).x);
            else
                res = vl_simplenn_modified(net,single(noisyCoeffsSub),[],[],...
                'mode','test',...
                'conserveMemory', 1);
                reconCoeffsSub = res(end-1).x;
            end
            
         
            reconCoeffs(yy:(yy+patchsize-1),xx:(xx+patchsize-1),:,ind_s:ind_e) =...
                reconCoeffs(yy:(yy+patchsize-1),xx:(xx+patchsize-1),:,ind_s:ind_e) + reconCoeffsSub;
        end
        
    end
    
end

wgtMap = repmat(wgtMap, [1, 1, size(reconCoeffs,3)]);
% if gpus > 0
%     reconCoeffs = gather(reconCoeffs);
% end

for tt = 1: size(reconCoeffs,4)
    reconCoeffs(:,:,:,tt) = reconCoeffs(:,:,:,tt)./wgtMap;
end

reconCoeffs = reconCoeffs + noisyCoeffs;

recon  = single(cnn_wavelet_recon(double(reconCoeffs./wgt),lv,dflt));
recon(recon < 0) = 0;

