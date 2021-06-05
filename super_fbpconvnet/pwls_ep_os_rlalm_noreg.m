 function [xs, info] = pwls_ep_os_rlalm_noreg(x, A, yi, jjj, varargin)


if nargin == 1 && streq(x, 'test'), ir_pwls_os_rlalm_test, return, end
if nargin < 4, help(mfilename), error(mfilename), end

% defaults
arg.nblock = 1;
arg.niter = 1;
arg.isave = [];
arg.userfun = @userfun_default;
arg.userarg = {};
arg.pixmax = inf;
arg.chat = false;
arg.wi = [];
arg.aai = [];
arg.rho = []; % default: decreasing rho
arg.alpha = 1.999;
arg.relax0 = 1;
arg.denom = [];
arg.scale_nblock = true; % traditional scaling
arg.update_even_if_denom_0 = true;
arg.xtrue = zeros(size(x));
arg.mask = ones(size(arg.xtrue));

arg = vararg_pair(arg, varargin);

arg.isave = iter_saver(arg.isave, arg.niter);


Ab = Gblock(A, arg.nblock);
% Ab = block_op(Ab, 'ensure'); % make it a block object (if not already)
nblock = block_op(Ab, 'n');
starts = subset_start(nblock);

cpu etic

wi = arg.wi;
if isempty(wi)
	wi = ones(size(yi));
end
if isempty(arg.aai)
	%arg.aai = reshape(sum(abs(Ab)'), size(yi)); % a_i = sum_j |a_ij|
    arg.aai = reshape(sum(Ab'), size(yi));
end

% check input sinogram sizes for OS
if (ndims(yi) ~= 2) || (size(yi,2) == 1 && nblock > 1)
	fail 'bad yi size'
end
if (ndims(wi) ~= 2) || (size(wi,2) == 1 && nblock > 1)
	fail 'bad wi size'
end

relax0 = arg.relax0(1);
if length(arg.relax0) == 1
	relax_rate = 0;
elseif length(arg.relax0) == 2
	relax_rate = arg.relax0(2);
else
	error relax
end

if length(arg.pixmax) == 2
	pixmin = arg.pixmax(1);
	pixmax = arg.pixmax(2);
elseif length(arg.pixmax) == 1
	pixmin = 0;
	pixmax = arg.pixmax;
else
	error pixmax
end

% likelihood denom, if not provided
denom = arg.denom;
if isempty(denom)
	denom = abs(Ab)' * col(arg.aai .* wi); % needs abs(Ab)
end
if ~arg.update_even_if_denom_0
	% todo: this may not work for LALM because "denom" appears in numerator!
	denom(denom == 0) = inf; % trick: prevents pixels where denom=0 being updated
end

% if isempty(R)
% 	pgrad = 0; % unregularized default
% 	Rdenom = 0;
% end

alpha = arg.alpha;
if alpha<1 || alpha>2
	fail 'alpha should be between 1 and 2'
end

rho = arg.rho;
if isempty(rho)
	rho = @(k) pi/(alpha*k) * sqrt(1 - (pi/(2*(alpha*k)))^2) * (k>1) + (k==1);
else
	rho = @(k) rho; % constant user-specified value
end

[nb na] = size(yi);

x = x(:);
np = length(x);
xs = zeros(np, length(arg.isave), 'single');
if any(arg.isave == 0)
	xs(:, arg.isave == 0) = single(x);
end

%info = zeros(niter,?); % do not initialize since size may change

% initialization
iblock = nblock;
ia = iblock:nblock:na;
li = Ab{iblock} * x;
li = reshape(li, nb, length(ia));
resid = wi(:,ia) .* (li - yi(:,ia));
if arg.scale_nblock
	scale = nblock; % traditional way
else
	scale = na / numel(ia); % alternative - untested
end
zeta = scale * Ab{iblock}' * resid(:);

g = rho(1) * zeta;
h = denom .* x - zeta;

info = struct('cost',[], 'RMSE',[], 'relE',[], 'SSIM', [], 'PSNR', []);
xtrue_msk = arg.xtrue(:);
SqrtPixNum = sqrt(numel(arg.mask));
% iterate
for iter = 1:arg.niter
	%ticker(mfilename, iter, arg.niter)
  if jjj == 1
     fprintf('%dth iteration of %d :\n', iter, arg.niter)
  end
  xold = x;
	relax = relax0 / (1 + relax_rate * (iter-1));

  if isempty(arg.xtrue) ~= true
     info.RMSE(:,iter) = norm(x - xtrue_msk) / SqrtPixNum;
     %disp(['RMSE = ', num2str(info.RMSE(:,iter))]);  
     info.SSIM(:,iter)= ssim(reshape(x,512,512), arg.xtrue);
     info.PSNR(:,iter) = computeRegressedSNR(reshape(x,512,512),arg.xtrue);
     %disp(['SSIM = ', num2str(info.SSIM(:,iter))]);  
  end  
  
	% loop over subsets
	for iset = 1:nblock
		k = nblock*(iter-1)+iset;

		num = rho(k) * (denom .* x - h) + (1-rho(k)) * g;
		den = rho(k) * denom;
% 		if ~isempty(R)
% 			num = num + R.cgrad(R, x);
% 			den = den + R.denom(R, x);
% 		end
		x = x - relax * num ./ den;
		x = max(x, pixmin);
		x = min(x, pixmax);

		iblock = starts(iset);
		ia = iblock:nblock:na;

		li = Ab{iblock} * x;
		li = reshape(li, nb, length(ia));
		resid = wi(:,ia) .* (li - yi(:,ia));

		if arg.scale_nblock
			scale = nblock; % traditional way
		else
			scale = na / numel(ia); % alternative - untested
		end

		zeta = scale * Ab{iblock}' * resid(:); % A' * W * (y - A*x)
		g = (rho(k) * (alpha * zeta + (1-alpha) * g) + g) / (rho(k)+1);
		h = alpha * (denom .* x - zeta) + (1-alpha) * h;

  end
  

	if any(arg.isave == iter)
		xs(:, arg.isave == iter) = single(x);
	end

  if arg.chat   
     df = .5 * sum(col(wi) .* (Ab * x - col(yi)).^2, 'double');
     fprintf('df = %g\n', df); 
     rp = R.penal(R, x);
     fprintf('rp = %g\n', rp); 
	   info.cost(:,iter) = df + rp;
  end


  info.relE(:,iter) = norm(xold - x) / norm(x);   
  %disp(['RelE = ', num2str( info.relE(:,iter))]);


  figure(20), imshow( embed(x, arg.mask), [800 1200]); 
end



% default user function.
% using this evalin('caller', ...) trick, one can compute anything of interest
function out = userfun_default(x, varargin)
chat = evalin('caller', 'arg.chat');
if chat
%	x = evalin('caller', 'x');
	printm('minmax(x) = %g %g', min(x), max(x))
end
out = cpu('etoc');


