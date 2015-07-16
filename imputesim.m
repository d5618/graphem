%IMPUTESIM  Estimate deleted values in simulated data.
%
%    Script to test imputation algorithms.
clear opt
path(path, '..')
%addpath('/Users/jeg/Documents/MATLAB/matlib/graphem');
rng('default');

n              = 55;                % sample size
p              = 17;                % number of variables
N              = 100;                % repetitions of simulation-imputation proc
%fmis         = [.1 .2 .3 .4];     % fraction of values missing       
fmis           = [.1, .2, .3]; 
nf             = length(fmis);     
ncv            = 0;                 % first ncv records have no missing values

% vector of mean values of data
mu             = 273.*ones(1, p);

% Matern covariance matrix of data
sigma          = 117;
scale          = 4;
nu             = 4;
d              = [0:p-1]; % distance = | i - j| in some matrix
C              = matern_cov(d, scale, sigma, nu);

% use low-rank approximation to covariance matrix to test truncation
% parameter choice
[V, d, r]      = peigs(C, min(n-1,p));
rt             = 11; %length(d);
C              = V(:,1:rt) * diag(d(1:rt)) * V(:,1:rt)';
R              = real(sqrtm(C));       

%  define graphical structure
adj = ones(p,p); k  = 3;
i = [1:p];
for j = 1:p
    adj(j,abs(j-i)>k) =0;
end

opt.adj       = adj;

% options for imputation algorithm
opt.disp       = 1;
opt.stagtol    = 1e-3;
opt.maxit      = 35;
opt.useggm     = 1 ;
opt.err_export = 1;
opt.nsample    = N;
rms_rerr       = zeros(N, nf);
rms_varerr     = zeros(N, nf);
err_errest     = zeros(N, nf);
cpustart       = cputime;

for jf=1:nf
  disp(sprintf('Fraction of values missing: %5.3f', fmis(jf)))

  for js=1:N    
    disp(sprintf('\n'))
    disp(sprintf('\tSimulation number %d:', js))
    
    % random Cholesky factor of covariance matrix
%     C          = zeros(p, p); 
%     for j=1:p
%       for k=j:p
%         R(j,k) = (2*rand-1)^(abs(j-k));
%       end
%     end
%     C          = R'*R;
    
    % generate random data
    xt         = repmat(mu, n, 1) + randn(n, p) * R;
    opt.Xcmp   = xt;
    
    % indices of missing values
    iperm      = randperm((n-ncv)*p) + ncv*p;
    nmis       = floor((n-ncv)*p*fmis(jf));
    % indices of missing values in transpose of data matrix (note that
    % Matlab indexes matrices in column order, but we need row order, so we
    % use transpose here and tranpose again later)
    indmist    = sort(iperm(1:nmis))';   
    [kmis, jmis] = ind2sub([p, n], indmist);
    indmis     = sub2ind([n, p], jmis, kmis);
    
    % delete values
    xd         = xt;
    xd(indmis) = repmat(NaN, nmis, 1);
    
    % impute values
    [xf, Mf, Cf, xerr_est, B, peff, kavlr, kmisr, iptrn, xres] = graphem_res(xd, opt);
    % [xf, Mf, Cf, xerr_est, B, peff, kavlr, kmisr, iptrn] = graphem(xd, opt);
    %[xfo, Mfo, Cfo, xerr_esto] = regem_26062013(xd, opt);

    % standard deviation of variables from complete dataset
    stdX                   = sqrt(diag(C));
    
    % error in error estimate
    xerr                   = xt(indmis)-xf(indmis);
    err_errest(js, jf)     = mean( (xerr_est(indmis)-abs(xerr))./stdX(kmis) );

    % rms relative error in imputed values
    rms_rerr(js,jf)        = norm( xerr./stdX(kmis) ) / sqrt(nmis);
   
    % rms relative error in diagonal elements of covariance matrices
    rms_varerr(js,jf)      = norm( (diag(Cf) - diag(C)) ./ diag(C)) / sqrt(nmis);
  end
end
cpuused = cputime - cpustart;
disp(sprintf('\nCPU time used: %9.1f s', cpuused))

% plot results
figure(1); clf
subplot(3,1,1)
hold on
for jf=1:nf
  plot(ones(N,1).*fmis(jf), rms_rerr(:,jf), '+')
end
axis([0 fmis(nf)+.05 0 2])
ylabel('Error in imputed values')
xlabel('Fraction of values missing')
grid on
disp(['Median error in imputed values: ', num2str(median(rms_rerr))])

subplot(3,1,2)
hold on
for jf=1:nf
  plot(ones(N,1).*fmis(jf), rms_varerr(:,jf), '+')
end
axis([0 fmis(nf)+.05 0 1])
ylabel('Error in variances')
xlabel('Fraction of values missing')
grid on
hold off
disp(['Median error in variances: ', num2str(median(rms_varerr))])

subplot(3,1,3)
hold on
for jf=1:nf
  plot(ones(N,1).*fmis(jf), err_errest(:,jf), '+')
end
axis([0 fmis(nf)+.05 -1 1])
ylabel('Estimated minus actual error')
xlabel('Fraction of values missing')
grid on
hold off
disp(['Median error in error estimate: ', num2str(median(err_errest))])
% hepta_figprint('imputesim_graphem')

% Analyze residuals
xp = repmat(xf,[1 1 N]) + xres;

figure(2); clf;
for js = 1:N
    tmp = xp(:,:,js);
    plot(tmp(indmis)); hold on;
end
plot(xf(indmis),'r-.','linewidth',2); 
plot(xt(indmis),'k-','linewidth',2); hold off;

figure(3); clf;
pss = 0;
for j = indmis'
    for js = 1:N
        tmp   = xres(:,:,js);
        x_hat(js) = tmp(j) + xf(j);
    end
    [f,xi] = ksdensity(x_hat,'function','pdf'); 
    plot(xi,f,'k','linewidth',2); hold on;
    x95 = ksdensity(x_hat,[0.025 0.975],'function','icdf');
    plot([x95(1) x95(1)],[0 2e-3],'r-.')
    plot([x95(2) x95(2)],[0 2e-3],'r-.')
    plot([xf(j) xf(j)],ylim,'k');
    plot([xt(j) xt(j)],ylim,'b'); hold off; pause;
    % plot(xf(j),'*'); hold off;
    
    if xf(j) >= x95(1) && xf(j) <= x95(2)
        pss = pss + 1;
    end
end
    

    
