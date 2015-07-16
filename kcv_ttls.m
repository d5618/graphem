function [ropt, xerr] = kcv_ttls(Vcv, dcv, Dcv, Mcv, kavlr, kmisr, ...
    outcv, X, kavlr_all, iptrn, rs, optreg)
%KCV_TTLS Selects truncation parameter for TTLS by K-fold cross-validation.
%   KCV_TTLS(Vcv, dcv, Dcv, Mcv, kavlr, kmisr, outcv, X, kavlr_all, iptrn,
%           ... rs, optreg)
%   returns the truncation parameter for TTLS that minimizes the squared
%   imputation error estimated by K-fold cross-validation. Only actually
%   available values in each validation fold are used to compute
%   cross-validated imputation errors.
%
%   KCV_TTLS takes the following input arguments:
%
%       Vcv{1:K}:   matrices of eigenvectors of the covariance matrix
%                   estimate for each of the K CV samples
%
%       dcv{1:K}:   associated eigenvalues for each CV sample
%
%       Dcv{1:K}:   vector of scale factors with which variables were scaled
%                   prior to them being passed to KCV_TTLS (e.g, these may be
%                   the diagonals of covariance matrix estimates for each
%                   CV sample if regularization is performed on correlation
%                   matrices)
%
%       Mcv{1:K}:   mean value vectors for each CV sample
%
%       kavlr:      indices of available values in current record (row of X)
%
%       kmisr:      indices of missing values in current record (row of X)
%
%       outcv{1:K}: left-out verification samples for each CV fold
%
%       X:          current estimate of data matrix X
%
%       kavlr_all:  cell array with indices of available values for each
%                   missigness pattern
%
%       iptrn(j):   index of pattern to which each row X(j,:) of data
%                   matrix belongs
%
%       rs:         vector of trunction parameters to be tried (if
%                   omitted, try all possible truncations)
%
%   The arguments Vcv, dcv, kavlr and kmisr correspond to input
%   parameters of PTTLS. outcv is output of KCVINDICES. iptrn is output of
%   MISSINGNESS_PATTERNS.
%
%   [ropt, xerr] = KCV_TTLS(...) returns the K-fold CV truncation parameter
%   ropt and a cross-validated rms error xerr of imputed values.

%   The rms error estimate xerr is corrected for bias using a heuristic of
%   Tibshirani and Tibshirani (Ann. Appl. Stat. 3, 2009).
%
%   if the optional regularixation parameter se_rule is set to 'true', the
% "one standard error rule" is applied as per
%  "Model selection and validation 2: Model assessment, more
%  cross-validation"
%  (http://www.stat.cmu.edu/~ryantibs/datamining/lectures/19-val2.pdf).


% process options
fopts      = fieldnames(optreg);

if find(strcmp('visual', fopts))
    visual = optreg.visual;
else
    visual    = false;
end

if strmatch('se_rule', fopts)
    se_rule = optreg.se_rule;
else
    se_rule = false;
end

% parameter K in K-fold cross-validation
Kcv    = length(Vcv);

if any(Kcv ~= [length(dcv), length(Dcv), length(Mcv), length(outcv)])
    error('Incompatible input arguments.')
end

% number of missing values in current record
nmis   = length(kmisr);

% vector of truncation parameters to be tried
rmax   = min([length(kavlr), length(dcv{1})-1]);
if ~exist('rs', 'var') || isempty(rs)
    rs = 1 : rmax;
end

if max(rs) > rmax
    irb  = find(rs <= rmax);
    rs   = rs(irb);
    warning('Maximum truncation level was too large and was lowered.')
end
nr     = length(rs);

% initialize cross-validation error to be accumulated
sek    = zeros(nr, nmis, Kcv); % squared CV imputation error for each fold
% and each missing value
nverk  = zeros(nmis, Kcv);     % number of actual verification values used in
% each fold and for each missing value
iroptk = zeros(1, Kcv);        % index of minimizer of CV error for k-th fold
nout   = 0;                    % number of rows of data matrix left out

for k=1:Kcv    % K-fold cross-validation loop
    % determine TTLS regression coefficients for current cross-validation
    % sample for all truncation paramters
    B  = pttls(Vcv{k}, dcv{k}, kavlr, kmisr, rs);
    
    % initialize verification-sample variables
    noutk = length(outcv{k}); % number of rows in current CV fold
    nout  = nout + noutk;
    Xmis  = zeros(noutk, nmis, nr);
    
    % re-center and re-scale left-out variables in the same way cv-sample
    % variables were adjusted
    Xoutcv = ( X(outcv{k}, :) - repmat(Mcv{k}, noutk, 1) ) ...
        ./ repmat(Dcv{k}', noutk, 1);
    
    % estimate left-out verification values for each
    % trial truncation parameter
    
    for ir=1:nr    % loop over truncation parameters
        Xmis(:, :, ir) = Xoutcv(:, kavlr) * B(:, :, ir);
        
        %restore original scaling of variables for computation of error
        Xmis(:, :, ir) = Xmis(:, :, ir) .* repmat(Dcv{k}(kmisr)', noutk, 1) ...
            + repmat(Mcv{k}(kmisr),  noutk, 1);
    end
   
    % compute deviation of imputed values from those actually available in
    % verification sample
    for j=1:noutk
        % loop over records (rows) j of left-out verification sample
        
        % first determine which of the imputed values in the verification
        % sample are actually available (rather than being imputed
        % themselves)
        kcv_avl = kavlr_all{iptrn(outcv{k}(j))};
        kcvind  = [];   % indices of available values for verification record j
        kmiscv  = [];   % corresponding indices of imputed values
        for ll = 1:nmis
            if any(kmisr(ll) == kcv_avl)
                kcvind   = [kcvind, kmisr(ll)];
                kmiscv   = [kmiscv, ll];
                nverk(ll, k) = nverk(ll, k) + 1;
            end
        end
        
        % sum up squared imputation error for each variable, scaled to D.
        for ir=1:nr      
            sek(ir, kmiscv, k) = sek(ir, kmiscv, k) ...
                + (X(outcv{k}(j), kcvind) - Xmis(j, kmiscv, ir)).^2;
        end
    end
    
    % mean squared error in cross-validation for k-th fold
    msek(:, :, k) = sek(:, :, k) ./ repmat(nverk(:, k)', nr, 1);
    
    % minimizer of squared error (summed over all missing values) for this
    % fold (used later for bias correction)
    [~, iroptk(k)] = min(sum(sek(:, :, k), 2), [], 1);
end

% total number of verification values for each missing value
nver   = sum(nverk, 2)';

if sum(nver) < 0.1 * nout * nmis
    warning(['Only ', num2str(min(nver)), ' verification values (', ...
        num2str(100*min(nver)/nout), '% of total) available in cross-validation sample. ', ...
        'Truncation parameter estimate may be inaccurate.'])
end

% If applicable, restrict subset of columns for KCV error calculation
cv_cols = ismember(kmisr,optreg.cv_cols);

% compute expected prediction error (EPE) over missing values
epe_c   = squeeze(mean(msek(:,cv_cols,:), 2)); % define prediction error metric
%epe_c = center(epe_c); %take out mean, which is different for every fold bc of non-stationarity.
epe_m  = mean(epe_c, 2);  % mean expected prediction error
epe_se = std(epe_c, 1, 2)/sqrt(Kcv); % standard error on EPE cf page 6 of R. Tibshirani's notes on KCV
upper = epe_m + epe_se;

% Determine truncation parameter as minimizer of squared imputation error.
% The minimum mse is the CV imputation error (which needs to be corrected
% for bias)
[~, iropt] = min(epe_m);

% If applicable, implement the one standard error rule
if se_rule
    iropt      = find(epe_m <= upper(iropt),1,'first');
else
    [~, iropt] = min(sum(sum(sek(:,cv_cols,:), 2), 3));
end

ropt       = rs(iropt);



if visual
    % % tapio's implementation
    [~, iropt2] = min(sum(sum(sek(:,cv_cols,:), 2), 3));
    ropt2       = rs(iropt2);
    figure(7), clf
    plot(rs,epe_c), hold on
    errorbar(rs,epe_m,epe_se,'ko');
    he = plot(rs,epe_m,'k-','linewidth',[2]);
    hm = plot(ropt,epe_m(iropt),'ro','linewidth',2);
    ht = plot(ropt2,epe_m(iropt2),'go','linewidth',2); hold off
    ttl = ['TTLS level choice by ', int2str(Kcv), '-fold cross-validation'];
    
    FontSize = 14; FontName = 'Palatino';
    hTitle=title(ttl);
    hYLabel=ylabel('CV generalization error');
    hXLabel=xlabel('Truncation Level');
    
    set(gca,'FontName','Times','FontSize', round(FontSize*0.71));
    set([hTitle, hXLabel, hYLabel], 'FontName' , FontName);
    set([hXLabel, hYLabel],'FontSize', round(FontSize*0.86));
    set(hTitle, 'FontSize', FontSize, 'FontWeight' , 'bold');
    set(gca,'Box', 'off', 'TickDir','out','TickLength',[.02 .02],'XMinorTick','on','YMinorTick','on', 'YGrid','on')
    set(gca,'XColor' , [.3 .3 .3], 'YColor', [.3 .3 .3], 'LineWidth', 1);
    
    legend([he hm ht],{'CV error \pm 1 SE','Chosen truncation level', 'CV MSE minimizer'},'location','best');
    legend boxoff
end

% ms imputation error over all verification folds
cverr      = sum(sek(iropt, :, :), 3) ./ nver;

% bias correction of CV estimate (according to Tibshirani and Tibshirani,
% Ann. Appl. Stat., 3 (2009))
cvbias     = zeros(1, nmis);
for k=1:Kcv
    cvbias  = cvbias + msek(iropt, :, k) - msek(iroptk(k), :, k);
end
cvbias     = cvbias / Kcv;
% Note that the bias is not necessarily positive for all variables here
% (unlike in the situation considered by TT2009). But its mean is usually
% positive.

% CV estimate of ms imputation error for each missing value
xerr       = sqrt(max(0, cverr + cvbias));

end

