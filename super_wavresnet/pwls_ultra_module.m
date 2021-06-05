function [xrla, info]=pwls_ultra_module(xini,xref,Ab,sino,wi,R,nIter,nOuterIter,denom,D_R,mu,SqrtPixNum,PatNum,ig)
xrla = xini;
xrla_msk = xrla(ig.mask);
stop_diff_tol = 1e-3; % HU
idx_old = ones([1,PatNum],'single');
info = struct('nOutIter',nOuterIter,'nIter',nIter,'xrla',[],'vIdx',[],'RMSE',[],'SSIM',[],'PSNR',[],'relE',[],'perc',[],'idx_change_perc',[],'cost',[]);
pixmax = inf;

AAA(1,1) = norm(xrla_msk - xref(ig.mask)) / SqrtPixNum;
info.RMSE = AAA(1,:);
AAA(2,1)= ssim(xrla, xref);
info.SSIM = AAA(2,:);
AAA(3,1)= computeRegressedSNR(xrla_msk, xref(ig.mask)); 
info.PSNR = AAA(3,:);
fprintf('PWLS-ULTRA Init: RMSE = %g, PSNR = %g, SSIM = %g,', AAA(1,1),AAA(3,1),AAA(2,1));

for ii=1:nOuterIter
    xold = xrla_msk;
     
   [xrla_msk, cost] = pwls_os_rlalm_l2normReg(xrla_msk, Ab, reshaper(sino, '2d'),  reshaper(wi, '2d'),...
        R, denom, D_R, mu,'pixmax', pixmax, 'chat', 1, 'alpha', 1.999, 'rho', [],'niter', nIter);
    AAA(1,ii+1) = norm(xrla_msk - xref(ig.mask)) / SqrtPixNum;
%     fprintf('RMSE = %g, ', AAA(1,ii));
    info.RMSE = AAA(1,:);
    AAA(2,ii+1)= ssim(xrla, xref);
%     fprintf('SSIM = %g\n', AAA(2,ii));
    info.SSIM = AAA(2,:);
    AAA(3,ii+1)= computeRegressedSNR(xrla_msk, xref(ig.mask)); 
    info.PSNR = AAA(3,:);
    fprintf('PWLS-ULTRA Iteration = %d:RMSE = %g, PSNR = %g, SSIM = %g,', ii,AAA(1,ii+1),AAA(3,ii+1),AAA(2,ii+1));
 
    [info.perc(:,ii),info.vIdx] = R.nextOuterIter();
    fprintf('perc = %g, ', info.perc(:,ii));

    info.idx_change_perc(:,ii) = nnz(idx_old - info.vIdx)/PatNum;
    fprintf('Idx Change Perc = %g, ', info.idx_change_perc(:,ii));
    idx_old = info.vIdx;

     info.cost(:,ii) = cost;
    info.relE(:,ii) =  norm(xrla_msk - xold) / SqrtPixNum;
    fprintf('relE = %g\n', info.relE(:,ii));
    if info.relE(:,ii) < stop_diff_tol
        break
    end

    xrla = ig.embed(xrla_msk);
%    figure(120), imshow(xrla, [800 1200]); drawnow;
end


