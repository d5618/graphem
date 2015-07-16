function [X, M, C, Xerr, W, adj] = GraphEM(X, options);
%REGEM 2.0 Imputation of missing values with regularized EM algorithm.
%         
%    [X, M, C, Xerr,W] = regem_2p0(X, OPTIONS) replaces missing values
%    (NaNs) in the data matrix X with imputed values. The algorithm
%    returns:
%
%       X,    the data matrix with imputed values substituted for NaNs,
%       M,    the estimated mean of X,
%       C,    the estimated covariance matrix of X,
%       Xerr, an estimated standard error of the imputed values.
%       W,    a structure of diagnostic outputs	:
%	     W{n}.avail    = available values for record n
%       W{n}.missing  = missing values for record n
%       W{n}.weights  = Regression matrix B (pm*pa)
%       W{n}.peff_ave = Effective # of parameters for that record
%	     W{n}.regpar 	 = regularization parameter
%
%    Missing values are imputed with a regularized expectation
%    maximization (EM) algorithm. In an iteration of the EM algorithm,
%    given estimates of the mean and of the covariance matrix are
%    revised in three steps. First, for each record X(i,:) with
%    missing values, the regression parameters of the variables with
%    missing values on the variables with available values are
%    computed from the estimates of the mean and of the covariance
%    matrix. Second, the missing values in a record X(i,:) are filled
%    in with their conditional expectation values given the available
%    values and the estimates of the mean and of the covariance
%    matrix, the conditional expectation values being the product of
%    the available values and the estimated regression
%    coefficients. Third, the mean and the covariance matrix are
%    re-estimated, the mean as the sample mean of the completed
%    dataset and the covariance matrix as the sum of the sample
%    covariance matrix of the completed dataset and an estimate of the
%    conditional covariance matrix of the imputation error.
%
%    In the regularized EM algorithm, the parameters of the regression
%    models are estimated by a regularized regression method. By
%    default, the parameters of the regression models are estimated by
%    an individual ridge regression for each missing value in a
%    record, with one regularization parameter (ridge parameter) per
%    missing value.  Optionally, the parameters of the regression
%    models can be estimated by a multiple ridge regression for each
%    record with missing values, with one regularization parameter per
%    record with missing values. The regularization parameters for the
%    ridge regressions are selected as the minimizers of the
%    generalized cross-validation (GCV) function. As another option,
%    the parameters of the regression models can be estimated by
%    truncated total least squares. The truncation parameter, a
%    discrete regularization parameter, is fixed and must be given as
%    an input argument. The regularized EM algorithm with truncated
%    total least squares is faster than the regularized EM algorithm
%    with with ridge regression, requiring only one eigendecomposition
%    per iteration instead of one eigendecomposition per record and
%    iteration.


%
%    Classic truncated total least squares regressions can be used to compute
%    initial values for EM iterations with ridge regressions, in which
%    the regularization parameter is chosen adaptively.
%
%    As default initial condition for the imputation algorithm, the
%    mean of the data is computed from the available values, mean
%    values are filled in for missing values, and a covariance matrix
%    is estimated as the sample covariance matrix of the completed
%    dataset with mean values substituted for missing
%    values. Optionally, initial estimates for the missing values and
%    for the covariance matrix estimate can be given as input
%    arguments.

%    Improvements in version 2.0 :
%	  -----------------------------
%     1) An adaptive choice of truncation parameter has been implemented for 
%     truncated total least squares (TTLS) and
%    principal component regression (PCR).
%    iTTLS is the TTLS equivalent of iRIDGE - its cousin iPCR the PCR
%    equivalent. iTTLS was found to outperform iRIDGE in terms of accuracy and
%    preservation of variance. In some contexts, iPCR achieves the same performance as
%    ITTLS at a much smaller cost. iTTLS is preferable in the paleocliamate
%    context, however. 

%	  Both require as input a range of truncation levels over which Bayesian
%    Model Averaging takes place (regpar = [kmin,kmax]. If the range is
%    unspecified, [1,min(n-1,p-1)} will be used. 

%     
%
%     2) The "block" option  enables block-by-block imputation when the same pattern 
%     of missing values is found repeated over several rows of the matrix. 
%     This can dramatically speed up computations (by an order of magnitude
%     in paleoclimate imputations). 
%      
%
%    The OPTIONS structure specifies parameters in the algorithm:
%
%     Field name         Parameter                                  Default
%
%     OPTIONS.regress    Regression procedure to be used:           'mridge'
%                        'mridge': multiple ridge regression (GCV)
%                        'iridge': individual ridge regressions (GCV)
%                        'bridge': BMA-based ridge regressions (BMA)
%                        'ttls'	: truncated total least squares
%	     						 'ittls': individual, adaptive TTLS (BMA)
%	   						 'ipcr':  individual, adaptive principal component
%	   						 regression (BMA)
%
%    OPTIONS.block       Uses block version of RegEM                     false
%                       (one regression per pattern of missing values): boolean
%                        The block version is considerably faster 
%                        when missing values have identical patterns 
%                        for several consdecutive row, as is often the 
%                        case with paleoclimate data. It is otherwise 
%                        marginally slower than looping over all rows
%                        of the matrix.
%
%     OPTIONS.stagtol    Stagnation tolerance: quit when            5e-3
%                        consecutive iterates of the missing
%                        values are so close that
%                          norm( Xmis(it)-Xmis(it-1) )
%                             <= stagtol * norm( Xmis(it-1) )
%
%     OPTIONS.maxit      Maximum number of EM iterations.           30
%
%     OPTIONS.inflation  Inflation factor for the residual          1
%                        covariance matrix. Because of the
%                        regularization, the residual covariance
%                        matrix underestimates the conditional
%                        covariance matrix of the imputation
%                        error. The inflation factor is to correct
%                        this underestimation. The update of the
%                        covariance matrix estimate is computed
%                        with residual covariance matrices
%                        inflated by the factor OPTIONS.inflation,
%                        and the estimates of the imputation error
%                        are inflated by the same factor.
%
%     OPTIONS.disp       Diagnostic output of algorithm. Set to     1
%                        zero for no diagnostic output.
%
%     OPTIONS.regpar     Regularization parameter.                  not set 
%                        -- For ridge regression, set regpar to 
%                        sqrt(eps) for mild regularization; leave 
%                        regpar unset for GCV selection of
%                        regularization parameters.
%                        -- For TTLS regression, set regpar to a 
%                        number for a fixed truncation. Set regpar
%                        to 'MDL', 'AIC', 'AICC', or 'NE08' for 
%                        different truncation choice criteria. If
%                        regpar is unset, NE08 is chosen as the
%                        default truncation choice criterion.
%                        (See PCA_TRUNCATION_CRITERIA for details.)
%
%                        For iTTLS or iPCR regression, regpar is a
%	                     vector specifiying the range of truncation
%	                     parameters ("models") to be explored. Averaging
%	                     takes place over all such models, above the
%	                     minimum quantile (quant_min). 
%
%     OPTIONS.quant_min  Lower bound of Occam's window: all models with 
%                        weights in the 1-quant_min range will be used to
%                        compute the Bayesian model average. (iPCR and
%                        iTTLS only)
%                        
%
%     OPTIONS.relvar_res Minimum relative variance of residuals.    5e-2
%                        From the parameter OPTIONS.relvar_res, a
%                        lower bound for the regularization
%                        parameter is constructed, in order to
%                        prevent GCV from erroneously choosing
%                        too small a regularization parameter.
%
%     OPTIONS.minvarfrac Minimum fraction of total variation in     0
%                        standardized variables that must be
%                        retained in the regularization.
%                        From the parameter OPTIONS.minvarfrac,
%                        an approximate upper bound for the
%                        regularization parameter is constructed.
%                        The default value OPTIONS.minvarfrac = 0
%                        essentially corresponds to no upper bound
%                        for the regularization parameter.
%
%     OPTIONS.Xmis0      Initial imputed values. Xmis0 is a         not set
%                        (possibly sparse) matrix of the same
%                        size as X with initial guesses in place
%                        of the NaNs in X.
%
%     OPTIONS.C0         Initial estimate of covariance matrix.     not set
%                        If no initial covariance matrix C0 is
%                        given but initial estimates Xmis0 of the
%                        missing values are given, the sample
%                        covariance matrix of the dataset
%                        completed with initial imputed values is
%                        taken as an initial estimate of the
%                        covariance matrix.
%
%     OPTIONS.Xcmp       Display the weighted rms difference        not set
%                        between the imputed values and the
%                        values given in Xcmp, a matrix of the
%                        same size as X but without missing
%                        values. By default, REGEM displays
%                        the rms difference between the imputed
%                        values at consecutive iterations. The
%                        option of displaying the difference
%                        between the imputed values and reference
%                        values exists for testing purposes.
%
%     OPTIONS.neigs      Number of eigenvalue-eigenvector pairs     not set
%                        to be computed for TTLS regression.
%                        By default, all nonzero eigenvalues and
%                        corresponding eigenvectors are computed.
%                        By computing fewer (neigs) eigenvectors,
%                        the computations can be accelerated, but
%                        the residual covariance matrices become
%                        inaccurate. Consequently, the residual
%                        covariance matrices underestimate the
%                        imputation error conditional covariance
%                        matrices more and more as neigs is
%                        decreased.
%
%    References:
%    [1] T. Schneider, 2001: Analysis of incomplete climate data:
%        Estimation of mean values and covariance matrices and
%        imputation of missing values. Journal of Climate, 14,
%        853--871.
%    [2] R. J. A. Little and D. B. Rubin, 1987: Statistical
%        Analysis with Missing Data. Wiley Series in Probability
%        and Mathematical Statistics. (For EM algorithm.)
%    [3] P. C. Hansen, 1997: Rank-Deficient and Discrete Ill-Posed
%        Problems: Numerical Aspects of Linear Inversion. SIAM
%        Monographs on Mathematical Modeling and Computation.
%        (For regularization techniques, including the selection of
%        regularization parameters.)
%    [4] Sima, D.M. and  Van Huffel, S. 2007 :
%        "Level choice in truncated total least squares",
%       Comput. Stat. & Data Analysis, 52, 1104-1118, doi:10.1016/j.csda.2007.05.015
%
%   History : adapted from v1.0 ( written by Tapio Schneider, CalTech)
%	  by Julien Emile-Geay, GaTech, 2007/2008
%                          USC,    2008/2010
%
%==================================================

error(nargchk(1, 2, nargin))     % check number of input arguments


if ndims(X) > 2,  error('X must be vector or 2-D array.'); end
% if X is a vector, make sure it is a column vector (a single variable)
if length(X)==numel(X)
   X = X(:);
end
[n, p]       = size(X);
% number of degrees of freedom for estimation of covariance matrix
dofC         = n - 1;            % use degrees of freedom correction

% ==============           process options        ========================
if nargin ==1 | isempty(options)
   fopts      = [];
else
   fopts      = fieldnames(options);
end

% initialize options structure for regression modules
optreg       = [];

regpar_given = 0;
if strmatch('regress', fopts)
   regress    = lower(options.regress);
   switch regress
      case {'mridge', 'iridge'}
         if strmatch('regpar', fopts) & ~isempty(options.regpar)
            regpar_given = 1;
            if ischar(options.regpar) 
               error('Regularization parameter must be a number')
            else
               optreg.regpar = options.regpar;
               regress       = 'mridge';
            end
         end
         
      case {'ttls'}
         trunc_max = min([options.regpar, n-1, p]);
         trunc_pars = [0: trunc_max - 1]; 
         if isempty(strmatch('regpar', fopts))
            trunc_criterion = 'mdl';    
         elseif ischar(options.regpar)
            trunc_criterion = lower(options.regpar);
         else
            regpar_given = 1;
            trunc = min([options.regpar, n-1, p]);
         end
         
         if strmatch('neigs', fopts)
            neigs  = options.neigs;
         else
            neigs  = min(n-1, p);
         end
         
      case {'ittls'}
         if strmatch('neigs', fopts)
            neigs  = options.neigs;
         else
            neigs  = min(n-1, p);
         end
         if strmatch('regpar', fopts) & numel(options.regpar) > 1 
            r  = [min(options.regpar):min([max(options.regpar), neigs])];
         else
            disp('Default TLS Truncation choice.')
            r	= neigs;
         end
         
      case {'mttls','ittlskcv'}
         if strmatch('neigs', fopts)
            neigs  = options.neigs;
         else
            neigs  = min(n-1, p);
         end
         
         if strmatch('cv_part', fopts)
            optreg.cv_part  = options.cv_part;
            kcv = length(optreg.cv_part);
         elseif strmatch('kcv', fopts)
            kcv = options.kcv;
            indices = crossvalind('Kfold',n,kcv);
            for k = 1:kcv
               optreg.cv_part{k} = (indices == k);
            end
         else
            error('MTTLS: cross-validation partition is underspecified')
         end
         
         if strmatch('regpar', fopts) & numel(options.regpar) > 1 
            r  = [min(options.regpar):min([max(options.regpar), neigs])];
            optreg.rvec = r;
         else
            disp('Default TLS Truncation choice.')
            r	= neigs;
         end
         
       case {'ggm'}
         %OK  
         
      otherwise
         error(['Unknown regression method ', regress])
         
   end
else
   regress    = 'mridge';
end


if strmatch('block', fopts)
   block    = options.block;
else
   block    = false;
end


if strmatch('stagtol', fopts)
   stagtol    = options.stagtol;
else
   stagtol    = 5e-3;
end

if strmatch('maxit', fopts)
   maxit      = options.maxit;
else
   maxit      = 30;
end

if strmatch('inflation', fopts)
   inflation  = options.inflation;
else
   inflation  = 1;
end

if strmatch('relvar_res', fopts)
   optreg.relvar_res = options.relvar_res;
else
   optreg.relvar_res = 5e-2;
end

if strmatch('minvarfrac', fopts)
   optreg.minvarfrac = options.minvarfrac;
else
   optreg.minvarfrac = 0;
end

h_given      = 0;
if strmatch('regpar', fopts) & ~isempty(options.regpar)
   h_given    = 1;
   optreg.regpar = options.regpar;
   if strmatch(regress, 'iridge')
      regress  = 'mridge';
   end
end


if strmatch('quant_min', fopts)
   optreg.quant_min = options.quant_min;
else
   optreg.quant_min = 0.1;
end


if strmatch('disp', fopts);
   dispon     = options.disp;
else
   dispon     = 1;
end

if strmatch('visual', fopts)
   optreg.visual = options.visual;
else
   optreg.visual = 0;
end

if strmatch('lonlat', fopts)
   optreg.lonlat = options.lonlat;
else
   optreg.lonlat = 0;
end

if strmatch('Xmis0', fopts);
   Xmis0_given= 1;
   Xmis0      = options.Xmis0;
   if any(size(Xmis0) ~= [n,p])
      error('OPTIONS.Xmis0 must have the same size as X.')
   end
else
   Xmis0_given= 0;
end

if strmatch('M0', fopts);
   M0_given   = 1;
   M0         = options.M0;
else
    M0_given = 0;
end


if strmatch('C0', fopts);
   C0_given   = 1;
   C0         = options.C0;
   if any(size(C0) ~= [p, p])
      error('OPTIONS.C0 has size incompatible with X.')
   end
else
   C0_given   = 0;
end

if strmatch('Xcmp', fopts);
   Xcmp_given = 1;
   Xcmp       = options.Xcmp;
   if any(size(Xcmp) ~= [n,p])
      error('OPTIONS.Xcmp must have the same size as X.')
   end
   sXcmp      = std(Xcmp);
else
   Xcmp_given = 0;
end

% Options for GGM    

if strmatch('useggm',fopts);
    useggm = options.useggm;
    if useggm
        opt_ggm = options_ggm(options);
        Xerr = NaN;
    else
        adj = NaN;
    end
else
    useggm = 0;
    adj = NaN;
end    


% =================           end options        =========================

% get indices of missing values and initialize matrix of imputed values
indmis       = find(isnan(X));
nmis         = length(indmis);
if nmis == 0
   warning('No missing value flags found.')
   return                                      % no missing values
end
[jmis,kmis]  = ind2sub([n, p], indmis);
Xmis         = sparse(jmis, kmis, NaN, n, p); % matrix of imputed values
Xerr         = sparse(jmis, kmis, Inf, n, p); % standard error imputed vals.

% Declare diagnostic array
if strcmp(regress,'ggm')
    fields = {'avail','missing','weights'};
else
    fields = {'avail','missing','weights','contrib','peff_ave','regpar'};
end


% for each row of X, assemble the column indices of the available
% values and of the missing values
kavlr        = cell(n,1);
kmisr        = cell(n,1);
for j=1:n
    kavlr{j}   = find(~isnan(X(j,:)));
    kmisr{j}   = find(isnan(X(j,:)));
end

% search for unique patterns
avail = double(~isnan(X)); 
[avail_uniq,I,J] = unique(avail,'rows','first');
np = size(avail_uniq,1); % number of patterns


if dispon
   disp(sprintf('\nREGEM:'))
   disp(sprintf('\tPercentage of values missing:      %5.2f', nmis/(n*p)*100))
   disp(sprintf('\tNumber of distinct patterns:       %4i', np))
   disp(sprintf('\tStagnation tolerance:              %9.2e', stagtol))
   disp(sprintf('\tMaximum number of iterations:      %3i', maxit))
   
   if (inflation ~= 1)
      disp(sprintf('\tResidual (co-)variance inflation:  %6.3f ', inflation))
   end
   
   if Xmis0_given & C0_given
      disp(sprintf(['\tInitialization with given imputed values and' ...
         ' covariance matrix.']))
   elseif C0_given
      disp(sprintf(['\tInitialization with given covariance' ...
         ' matrix.']))
   elseif Xmis0_given
      disp(sprintf(['\tInitialization with given imputed values.']))
   else
      disp(sprintf('\tInitialization of missing values by mean substitution.'))
   end
   
   switch regress
      case 'mridge'
         disp(sprintf('\tOne multiple ridge regression per pattern:'))
         disp(sprintf('\t==> one regularization parameter per pattern.'))
      case 'iridge'
         disp(sprintf('\tOne individual ridge regression per missing value:'))
         disp(sprintf('\t==> one regularization parameter per missing value.'))
      case 'ttls'
         if regpar_given
            disp(sprintf('\tFixed truncation parameter:     %5i', trunc))
         else
            r = trunc_pars;
            disp(sprintf(['\tTruncation choice criterion: ', upper(trunc_criterion),', r in [%d;%d]'],r(1),r(end)))
         end
      case {'ittls'}
         disp(sprintf('\tOne individual TTLS regression per missing value.'))
         disp(sprintf('\t==> Truncation range:  [%d;%d]', r(1),r(end)))
      case {'mttls'}
         disp(sprintf('\tOne total least squares regression per pattern'))
         disp(sprintf('\t==> Truncation range:  [%d;%d]', r(1),r(end)))
      case {'ittlskcv'}
         disp(sprintf('\tOne total least squares regression per missing value'))
         disp(sprintf('\t==> Truncation range:  [%d;%d]', r(1),r(end)))
      case 'ipcr'
         disp(sprintf('\tOne individual PC regression per missing value.'))
         disp(sprintf('\t==> Truncation range:  [%d;%d]', r(1),r(end)))
   end
   
   if useggm
       disp('Using GGM') 
   end
   
   
   if Xcmp_given
     if useggm
           disp(sprintf('\n\tIter \tmean(peff) \t|D(Xmis)| \t|D(Xmis)|/|Xmis| \t Sparsity level'))
       else
           disp(sprintf('\n\tIter \tmean(peff) \t|D(Xmis)| \t|D(Xmis)|/|Xmis|'))
      end
   else
       if useggm
           disp(sprintf('\n\tIter \tmean(peff) \t|D(Xmis)| \t|D(Xmis)|/|Xmis| \t Sparsity level'))
       else
           disp(sprintf('\n\tIter \tmean(peff) \t|D(Xmis)| \t|D(Xmis)|/|Xmis|'))
      end
   end
end

% initial estimates of missing values
if Xmis0_given
    % substitute given guesses for missing values
   X(indmis)  = Xmis0(indmis);
   [X, M]     = center(X);        % center data to mean zero
else
    if M0_given
        M = M0;
        X = X - repmat(M,n,1);
        X(indmis)  = zeros(nmis, 1);   % fill missing entries with zeros
    else
        [X, M]     = center(X);        % center data to mean zero
        X(indmis)  = zeros(nmis, 1);   % fill missing entries with zeros
    end
end

if C0_given
   C          = C0;
else
   C          = X'*X / dofC;      % initial estimate of covariance matrix
   if(strcmp(regress, 'ggm'))
       C = C + 0.1*eye(p);
   end
end

if(strcmp(regress, 'ggm'))
    O = inv(C);
end


if strcmpi(regress,'mttls') | strcmpi(regress,'ittlskcv');
   mu_i = cell(1,kcv);
   for i = 1:kcv    
      test_cv{i}  = find(optreg.cv_part{i});
      train_cv{i} = setdiff([1:n],test_cv{i});
      X_cv{i} = X(train_cv{i},:);
      X_cv_test{i} = X(test_cv{i},:);
      
      % Center training and validation data
      [X_cv{i},mu_i{i}] = center(X_cv{i});
      [X_cv_test{i},mu_i_test{i}] = center(X_cv_test{i});
      
      % Scale test data
      n_test = size(X_cv_test{i},1);
      D = std(X_cv_test{i});
      % scale variables to unit variance (as for complete sample)
      const        = (abs(D) < eps);   % test for constant variables
      nconst       = ~const;
      if sum(const) ~= 0             % do not scale constant variables
         D        = D .* nconst + 1*const;
      end
      X_cv_test{i} = X_cv_test{i} ./ repmat(D, n_test, 1);
      
      dofCV(i) = size(X_cv{i},1)-1;
      C_cv{i} = X_cv{i}'*X_cv{i} / dofCV(i);
   end
   optreg.test  = test_cv; 
   optreg.train = train_cv;
end

it           = 0;
rdXmis       = Inf;
while (it < maxit & rdXmis > stagtol)
   it         = it + 1;
   
   % initialize for this iteration ...
   CovRes     = zeros(p,p);       % ... residual covariance matrix
   peff_ave   = 0;                % ... average effective number of variables
   
   % scale variables to unit variance
   D          = sqrt(diag(C));
   const      = (abs(D) < eps);   % test for constant variables
   nconst     = ~const;
   if sum(const) ~= 0             % do not scale constant variables
      D        = D .* nconst + 1*const;
   end
   X          = X ./ repmat(D', n, 1);
   % correlation matrix
   C          = C ./ repmat(D', p, 1) ./ repmat(D, 1, p);
   if useggm & strcmpi(regress, 'ggm') 
       O          = O .* repmat(D', p, 1) .* repmat(D, 1, p);
   end
      
   
   if strcmpi(regress, 'ttls');
      % compute eigendecomposition of correlation matrix
      [V, d]   = peigs_n(C);
      % compute truncation selection criteria if needed
      if strcmpi(regress,'ttls') & ~regpar_given
         [wk85, ne08] = pca_truncation_criteria(d, p, r, n);
      end
   elseif strcmpi(regress, 'mttls') | strcmpi(regress, 'ittlskcv');
      % compute eigendecomposition of correlation matrix
      [V, d]   = peigs_n(C);
     
      % Compute eigendecompositions withholding cross-validation partitions
      [C_cv, X_cv, V_cv, d_cv, D_cv] = eig_cv(C_cv, X_cv, dofCV, p, kcv);
      optreg.eigvec    = V_cv;
      optreg.eigval    = d_cv; 
        
      % define residual covariance matrix for cross-validation estimates
      CovRes_cv    = zeros(p,p,kcv); 
   end
   
   if block  % If block version is chosen...
      for j=1:np             % cycle over patterns
         % Fill in the pattern cell array
         avail_m = avail - repmat(avail_uniq(j,:),n,1);
         pattern{j} = find(std(avail_m,0,2) == 0)';
         jp     = min(pattern{j});             % position of this pattern
         pm     = length(kmisr{jp});  % number of missing values in this pattern
         if pm > 0
            mp     = len(pattern{j});    % number of rows matching this pattern
            pa     = p - pm;           % number of available values in this pattern
            
            % regression of missing variables on available variables
            switch regress
               case 'mridge'
                  % one multiple ridge regression per patterm (GCV)
                  [B, S, h, peff]   = mridge(C(kavlr{jp},kavlr{jp}), ...
                     C(kmisr{jp},kmisr{jp}), ...
                     C(kavlr{jp},kmisr{jp}), n-1, optreg);
                  
                  peff_ave = peff_ave + peff*pm*mp/nmis;  % add up eff. number of variables
                  dofS     = dofC - peff;              % residual degrees of freedom
                  
                  % inflation of residual covariance matrix
                  S        = inflation * S;
                  
                  % bias-corrected estimate of standard error in imputed values
                  Xerr(pattern{j}, kmisr{jp}) = repmat(dofC/dofS * sqrt(diag(S))',[mp 1]);
                  
                  % Save diagnostic output
                  if (nargout >= 5)
                     BXa = X(pattern{j}, kavlr{jp})* B; 
                     w={kavlr{jp},kmisr{jp}, B, BXa, mean(peff),h};
                     W(pattern{j}) = cell2struct(w, fields,2);
                  end
                  
               case 'iridge'
                  % one individual ridge regression per patterm (GCV)
                  [B, S, h, peff]   = iridge(C(kavlr{jp},kavlr{jp}), ...
                     C(kmisr{jp},kmisr{jp}), ...
                     C(kavlr{jp},kmisr{jp}), n-1, optreg);
                  
                  peff_ave = peff_ave + sum(peff)*mp/nmis; % add up eff. number of variables
                  dofS     = dofC - peff;               % residual degrees of freedom
                  
                  % inflation of residual covariance matrix
                  S        = inflation * S;
                  
                  % bias-corrected estimate of standard error in imputed values
                  
                  Xerr(pattern{j}, kmisr{jp}) = repmat(( dofC * sqrt(diag(S)) ./ dofS)',[mp 1]);
                  
                  % Save diagnostic output
                  if (nargout >= 5)
                     BXa = X(pattern{j}, kavlr{jp})* B; 
                     w={kavlr{jp},kmisr{jp}, B, BXa, mean(peff),h};
                     W(pattern{j}) = cell2struct(w, fields,2);
                  end
%                case 'bridge'
%                   % one individual ridge regression per patterm (BMA)
%                   [B, S, h, peff]   = bridge(C(kavlr{jp},kavlr{jp}), ...
%                      C(kmisr{jp},kmisr{jp}), ...
%                      C(kavlr{jp},kmisr{jp}), n-1, optreg);
%                   
%                   peff_ave = peff_ave + sum(peff)*mp/nmis; % add up eff. number of variables
%                   dofS     = dofC - peff;               % residual degrees of freedom
%                   
%                   % inflation of residual covariance matrix
%                   S        = inflation * S;
%                   
%                   % bias-corrected estimate of standard error in imputed values
%                   
%                   Xerr(pattern{j}, kmisr{jp}) = repmat(( dofC * sqrt(diag(S)) ./ dofS)',[mp 1]);
%                   
%                   % Save diagnostic output
%                   if (nargout >= 5)
%                      BXa = X(pattern{j}, kavlr{jp})* B; 
%                      w={kavlr{jp},kmisr{jp}, B, BXa, mean(peff),h};
%                      W(pattern{j}) = cell2struct(w, fields,2);
%                   end
                              
%                case 'ipcr'
%                   % one individual PC regression per pattern per missing
%                   % value (based solely on factorizations of C)
%                   [B, S, truncv, peff]   = ipcr(C(kavlr{jp},kavlr{jp}), ...
%                      C(kmisr{jp},kmisr{jp}), ...
%                      C(kavlr{jp},kmisr{jp}), n-1, optreg.quant_min);
%                   
%                   peff_ave = peff_ave + sum(peff)*mp/nmis; % add up eff. number of variables
%                   dofS     = dofC - peff;               % residual degrees of freedom
%                   
%                   % inflation of residual covariance matrix
%                   S        = inflation * S;
%                   
%                   % bias-corrected estimate of standard error in imputed values
%                   Xerr(pattern{j}, kmisr{jp}) = repmat(( dofC * sqrt(diag(S)) ./ dofS)',[mp 1]);
%                   
%                   % Save diagnostic output
%                   if (nargout >= 5)
%                       BXa = X(pattern{j}, kavlr{jp})* B;
%                       w   = {kavlr{jp}, kmisr{jp}, B, BXa, mean(peff), truncv};
%                       W(pattern{j}) = cell2struct(w, fields,2);
%                   end
                  
               case 'ttls'     
                  % truncated total least squares      
                  if ~regpar_given
                     [~, imin]     = eval(['min(', trunc_criterion, ')']);
                     trunc           = trunc_pars(imin);
                  end
                  
                  peff_ave = peff_ave + trunc*pm*mp/nmis;  % add up eff. number of variables
                  
                  % truncated total least squares with fixed truncation parameter
                  [B, S]   = pttls(V, d, kavlr{jp}, kmisr{jp}, trunc);
                  dofS    = dofC - trunc;         % residual degrees of freedom
                  
                  % inflation of residual covariance matrix
                  S        = inflation * S;
                  % bias-corrected estimate of standard error in imputed values
                  Xerr(pattern{j}, kmisr{jp}) = repmat(dofC/dofS * sqrt(diag(S))',[mp 1]);
                  % Save diagnostic output
                  if (nargout >= 5)
                      BXa = X(pattern{j}, kavlr{jp})* B; 
                      w   = {kavlr{jp}, kmisr{jp}, B, BXa, trunc, trunc};
                      W(pattern{j}) = cell2struct(w, fields,2);
                  end
                  
                  
               case 'ittls'
                  % truncated total least squares with adaptive truncation parameter
                  % One truncation parameter per patterm per missing value
                  [B, S, trunc, peff] = ittls(C, kavlr{jp}, kmisr{jp}, n, r, optreg);
                  
                  peff_ave = peff_ave + sum(peff)*mp/nmis; % add up eff. number of variables
                  dofS     = dofC - peff;
                  
                  % inflation of residual covariance matrix
                  S        = inflation * S;
                  
                  % bias-corrected estimate of standard error in imputed values
                  Xerr(pattern{j}, kmisr{jp}) = repmat(( dofC * sqrt(diag(S)) ./ dofS)',[mp 1]);
                  
                  % Save diagnostic output
                  if (nargout >= 5)
                     BXa = X(pattern{j}, kavlr{jp})* B; 
                     w={kavlr{jp}, kmisr{jp}, B, BXa, mean(peff), trunc};
                     W(pattern{j}) = cell2struct(w, fields,2);
                  end
                  
               case 'mttls'
                  % truncated total least squares with adaptive truncation choice
                  % one truncation parameter per record. K-fold cross-validation
                  
                  
                  [B, S, trunc, peff, S_cv] = mttls(X_cv_test, V, d, kavlr{jp}, kmisr{jp}, optreg); 
                  
                  
                  peff_ave = peff_ave + trunc*pm*mp/nmis;  % add up eff. number of variables
                  %peff_ave = peff_ave + trunc*mp/nmis; 
                  
                  dofS    = dofC - trunc;         % residual degrees of freedom
                  
                  %  assemble residual CV covariance matrix 
                  CovRes_cv(kmisr{jp}, kmisr{jp},:) = CovRes_cv(kmisr{jp}, kmisr{jp},:) + S_cv;
                                   
                  % inflation of residual covariance matrix
                  S        = inflation * S;
                  
                  % bias-corrected estimate of standard error in imputed values
                  Xerr(pattern{j}, kmisr{jp}) = repmat(( dofC * sqrt(diag(S)) ./ dofS)',[mp 1]);
                  
                  % Save diagnostic output
                  if (nargout >= 5)
                     BXa = X(pattern{j}, kavlr{jp})* B; 
                     w={kavlr{jp}, kmisr{jp}, B, BXa, peff_ave, trunc};
                     W(pattern{j}) = cell2struct(w, fields,2);
                  end
                  
                  case 'ittlskcv'
                  % truncated total least squares with adaptive truncation choice
                  % one truncation parameter per record. K-fold cross-validation
                  
                  
                  [B, S, trunc, peff, S_cv] = ittlskcv(X_cv_test, V, d, kavlr{jp}, kmisr{jp}, optreg); 
                  
                  
                  peff_ave = peff_ave + trunc*pm*mp/nmis;  % add up eff. number of variables
                  %peff_ave = peff_ave + trunc*mp/nmis; 
                  
                  dofS    = dofC - trunc;         % residual degrees of freedom
                  
                  %  assemble residual CV covariance matrix 
                  CovRes_cv(kmisr{jp}, kmisr{jp},:) = CovRes_cv(kmisr{jp}, kmisr{jp},:) + S_cv;
                                   
                  % inflation of residual covariance matrix
                  S        = inflation * S;
                  
                  % bias-corrected estimate of standard error in imputed values
                  Xerr(pattern{j}, kmisr{jp}) = repmat(( dofC * sqrt(diag(S)) ./ dofS)',[mp 1]);
                  
                  % Save diagnostic output
                  if (nargout >= 5)
                     BXa = X(pattern{j}, kavlr{jp})* B; 
                     w={kavlr{jp}, kmisr{jp}, B, BXa, peff_ave, trunc};
                     W(pattern{j}) = cell2struct(w, fields,2);
                  end
                  
                case 'ggm'      
                    [B,S] = ggm(C,kavlr{jp},kmisr{jp});
                    % inflation of residual covariance matrix
                    S        = inflation * S;
                    w={kavlr{jp}, kmisr{jp}, B};
                    W(pattern{j}) = cell2struct(w, fields,2);
            end

            % missing value estimates
            Xmis(pattern{j}, kmisr{jp})  = X(pattern{j}, kavlr{jp}) * B;
            S = (S+S.')/2;
            Xmis(pattern{j}, kmisr{jp}) = Xmis(pattern{j}, kmisr{jp}) + mvnrnd(zeros(1,length(kmisr{jp})),S,length(pattern{j}));

            % add up contribution from residual covariance matrices
            CovRes(kmisr{jp}, kmisr{jp}) = CovRes(kmisr{jp}, kmisr{jp}) + mp*S;
         end
      end
      
   else   %  if block version is not desirable, loop over records as in version 1
      
      for j=1:n              % cycle over records
         pm       = length(kmisr{j}); % number of missing values in this record
         if pm > 0
            pa     = p - pm;           % number of available values in this record
            
            % regression of missing variables on available variables
            switch regress
               case 'mridge'
                  % one multiple ridge regression per record
                  [B, S, h, peff]   = mridge(C(kavlr{j},kavlr{j}),C(kmisr{j},kmisr{j}), C(kavlr{j},kmisr{j}), n-1, optreg);
                  
                  peff_ave = peff_ave + peff*pm/nmis;  % add up eff. number of variables
                  dofS     = dofC - peff;              % residual degrees of freedom
                  
                  % inflation of residual covariance matrix
                  S        = inflation * S;
                  
                  % bias-corrected estimate of standard error in imputed values
                  Xerr(j, kmisr{j}) = dofC/dofS * sqrt(diag(S))';
                  
                  % Save diagnostic output
                  if (nargout >= 5)
                      BXa  = X(j, kavlr{j})* B;
                      w    = {kavlr{j},kmisr{j},B,BXa,peff_ave,h};
                      W(j) = cell2struct(w, fields,2);
                  end
                  
               case 'iridge'
                  % one individual ridge regression per missing value per record
                  [B, S, h, peff]   = iridge(C(kavlr{j},kavlr{j}), ...
                     C(kmisr{j},kmisr{j}), ...
                     C(kavlr{j},kmisr{j}), n-1, optreg);
                  
                  peff_ave = peff_ave + sum(peff)/nmis; % add up eff. number of variables
                  dofS     = dofC - peff;               % residual degrees of freedom
                  
                  % inflation of residual covariance matrix
                  S        = inflation * S;
                  
                  % bias-corrected estimate of standard error in imputed values
						
                  Xerr(j, kmisr{j}) = ( dofC * sqrt(diag(S)) ./ dofS)';
                  
                  % Save diagnostic output
                  if (nargout >= 5)
                     BXa  = X(j, kavlr{j})* B;
                     w    = {kavlr{j},kmisr{j},B,BXa,mean(peff),h};
                     W(j) = cell2struct(w, fields,2);
                  end
                  
%                case 'bridge'
%                   % one individual ridge regression per missing value per record
%                   [B, S, h, peff]   = bridge(C(kavlr{j},kavlr{j}), ...
%                      C(kmisr{j},kmisr{j}), ...
%                      C(kavlr{j},kmisr{j}), n-1, optreg);
%                   
%                   peff_ave = peff_ave + sum(peff)/nmis; % add up eff. number of variables
%                   dofS     = dofC - peff;               % residual degrees of freedom
%                   
%                   % inflation of residual covariance matrix
%                   S        = inflation * S;
%                   
%                   % bias-corrected estimate of standard error in imputed values
%                   Xerr(j, kmisr{j}) = ( dofC * sqrt(diag(S)) ./ dofS)';
%                   
%                   % Save diagnostic output
%                   if (nargout >= 5)
%                      BXa  = X(j, kavlr{j})* B;
%                      w    = {kavlr{j},kmisr{j},B,BXa,mean(peff),h};
%                      W(j) = cell2struct(w, fields,2);
%                   end
                                    
%                case 'ipcr'
%                   % one individual PC regression per record per
%                   % missing value 
%                   [B, S, truncv, peff]   = ipcr_nofudge(C(kavlr{j},kavlr{j}), ...
%                      C(kmisr{j},kmisr{j}), ...
%                      C(kavlr{j},kmisr{j}), n-1, n, optreg.quant_min);
%                   
%                   peff_ave = peff_ave + sum(peff)/nmis; % add up eff. number of variables
%                   dofS     = dofC - peff;               % residual degrees of freedom
%                   
%                   % inflation of residual covariance matrix
%                   S        = inflation * S;
%                   
%                   % bias-corrected estimate of standard error in imputed values
%                   Xerr(j, kmisr{j}) = ( dofC * sqrt(diag(S)) ./ dofS)';
%                   
%                   % Save diagnostic output
%                   if (nargout >= 5)
%                      BXa  = X(j, kavlr{j})* B;
%                      w    = {kavlr{j},kmisr{j},B,BXa,mean(peff),truncv};
%                      W(j) = cell2struct(w, fields,2);
%                   end
                  
               case 'ttls'
                  % truncated total least squares
                  if ~regpar_given
                     [~, imin]     = eval(['min(', trunc_criterion, ')']);
                     trunc           = trunc_pars(imin);
                  end
                  peff_ave = peff_ave + trunc*pm/nmis;  % add up eff. number of variables
                  
                  [B, S]   = pttls(V, d, kavlr{j}, kmisr{j}, trunc);
                  dofS    = dofC - trunc;         % residual degrees of freedom
                  
                  % inflation of residual covariance matrix
                  S        = inflation * S;
                  % bias-corrected estimate of standard error in imputed values
                  
                  Xerr(j, kmisr{j}) = dofC/dofS * sqrt(diag(S))';
                  % Save diagnostic output
                  if (nargout >= 5)
                     BXa  = X(j, kavlr{j})* B;
                     w    = {kavlr{j}, kmisr{j}, B, BXa, peff_ave, trunc};
                     W(j) = cell2struct(w, fields,2);
                  end
                  
                  
               case 'ittls'
                  % truncated total least squares with adaptive truncation parameter
                  % One truncation parameter per record per missing value
                  [B, S, trunc, peff] = ittls(C, kavlr{j}, kmisr{j}, n, r, optreg);
                  
                  peff_ave = peff_ave + sum(peff)/nmis; % add up eff. number of variables
                  dofS     = dofC - peff;
                  
                  % inflation of residual covariance matrix
                  S        = inflation * S;
                  
                  % bias-corrected estimate of standard error in imputed values
                  Xerr(j, kmisr{j}) = ( dofC * sqrt(diag(S)) ./ dofS)';
                  
                  % Save diagnostic output
                  if (nargout >= 5)
                     BXa  = X(j, kavlr{j})* B;
                     w	  = {kavlr{j}, kmisr{j}, B, BXa, mean(peff), trunc};
                     W(j) = cell2struct(w, fields,2);
                  end
                  
                case 'mttls'
                  % truncated total least squares with adaptive truncation parameter
                  % One truncation parameter per record       
                  
                  [B, S, trunc, peff, S_cv] = mttls(X_cv_test, V, d, kavlr{j}, kmisr{j}, optreg); 
     
                  
                  peff_ave = peff_ave + trunc*pm/nmis;  % add up eff. number of variables
                  dofS    = dofC - trunc;         % residual degrees of freedom
                  
                  %  assemble residual CV covariance matrix 
                  CovRes_cv(kmisr{j}, kmisr{j},:) = CovRes_cv(kmisr{j}, kmisr{j},:) + S_cv;
                  
                  % inflation of residual covariance matrix
                  S        = inflation * S;
                  
                  % bias-corrected estimate of standard error in imputed values
                  Xerr(j, kmisr{j}) = ( dofC * sqrt(diag(S)) ./ dofS)';
                  
                  % Save diagnostic output
                  if (nargout >= 5)
                     BXa  = X(j, kavlr{j})* B;
                     w	  = {kavlr{j}, kmisr{j}, B, BXa, peff_ave, trunc};
                     W(j) = cell2struct(w, fields,2);
                  end
                  
                  case 'ittlskcv'
                  % truncated total least squares with adaptive truncation parameter
                  % One truncation parameter per record       
                  
                  [B, S, trunc, peff, S_cv] = ittlskcv(X_cv_test, V, d, kavlr{j}, kmisr{j}, optreg); 
     
                  
                  peff_ave = peff_ave + trunc*pm/nmis;  % add up eff. number of variables
                  dofS    = dofC - trunc;         % residual degrees of freedom
                  
                  %  assemble residual CV covariance matrix 
                  CovRes_cv(kmisr{j}, kmisr{j},:) = CovRes_cv(kmisr{j}, kmisr{j},:) + S_cv;
                  
                  % inflation of residual covariance matrix
                  S        = inflation * S;
                  
                  % bias-corrected estimate of standard error in imputed values
                  Xerr(j, kmisr{j}) = ( dofC * sqrt(diag(S)) ./ dofS)';
                  
                  % Save diagnostic output
                  if (nargout >= 5)
                     BXa  = X(j, kavlr{j})* B;
                     w	  = {kavlr{j}, kmisr{j}, B, BXa, peff_ave, trunc};
                     W(j) = cell2struct(w, fields,2);
                  end
                  
                case 'ggm'
                    [B,S] = ggm(C,kavlr{j},kmisr{j});
                    % inflation of residual covariance matrix
                    S        = inflation * S;
                    w	  = {kavlr{j}, kmisr{j}, B};
                    W(j) = cell2struct(w, fields,2);
            end
            % missing value estimates
            Xmis(j, kmisr{j})   = X(j, kavlr{j}) * B;
            Xmis(j, kmisr{j}) = Xmis(j, kmisr{j}) + mvnrnd(zeros(1,length(kmisr{j})),S,1);

            % add up contribution from residual covariance matrices
            CovRes(kmisr{j}, kmisr{j}) = CovRes(kmisr{j}, kmisr{j}) + S;
         end  
      end
   end

   % rescale variables to original scaling
   X          = X .* repmat(D', n, 1);
   Xerr       = Xerr .* repmat(D', n, 1);
   Xmis       = Xmis .* repmat(D', n, 1);
   C          = C .* repmat(D', p, 1) .* repmat(D, 1, p);
   CovRes     = CovRes .* repmat(D', p, 1) .* repmat(D, 1, p);
   
   
         %disp(['Cov residual norm = ',num2str(norm(CovRes,'fro'))]);
   
   % rms change of missing values
   dXmis      = norm(Xmis(indmis) - X(indmis)) / sqrt(nmis);
   
   % relative change of missing values
   nXmis_pre  = norm(X(indmis) + M(kmis)') / sqrt(nmis);
   if nXmis_pre < eps
      rdXmis   = Inf;
   else
      rdXmis   = dXmis / nXmis_pre;
   end
   
   % update data matrix X
   X(indmis)  = Xmis(indmis);
   
   % re-center data and update mean
   [X, Mup]   = center(X);                  % re-center data
   M          = M + Mup;                    % updated mean vector
   
   % update covariance matrix estimate
   if useggm
       [C,O,sp_level,adj,opt_ggm] = Sigma_G(X,CovRes,dofC,opt_ggm);
       O = sparse(O);
   else
       C          = (X'*X + CovRes)/dofC;
   end
   
   
   if strcmpi(regress,'mttls') | strcmpi(regress,'ittlskcv')
      for i = 1:kcv
          X_cv{i}          = X(train_cv{i},:); % CHECK SCALING
         [X_cv{i},mu_i{i}] = center(X_cv{i});
         
         X_cv_test{i} = X(test_cv{i},:);
         [X_cv_test{i},mu_i_test{i}] = center(X_cv_test{i});
         % Scale test data
         n_test = size(X_cv_test{i},1);
         D = std(X_cv_test{i});
         const        = (abs(D) < eps);   % test for constant variables
         nconst       = ~const;
         if sum(const) ~= 0             % do not scale constant variables
            D        = D .* nconst + 1*const;
         end
         X_cv_test{i} = X_cv_test{i} ./ repmat(D, n_test, 1);
         
         CovRes_cv(:,:,i) = squeeze(CovRes_cv(:,:,i)) .* repmat(D_cv{i}', p, 1) .* repmat(D_cv{i}, 1, p);    
         
         % Compute covariance matrix
         if useggm
             [C_cv{i},~,~,opt_ggm] = Sigma_G(X_cv{i},CovRes_cv(:,:,i),dofCV(i),opt_ggm);
         else
             C_cv{i} = (X_cv{i}'* X_cv{i} + CovRes_cv(:,:,i))/dofCV(i);
         end
      end
   end
   
   if dispon
      if Xcmp_given
         % imputed values in original scaling
         Xmis(indmis) = X(indmis) + M(kmis)';
         % relative error of imputed values (relative to values in Xcmp)
         dXmis        = norm( (Xmis(indmis)-Xcmp(indmis))./sXcmp(kmis)' ) ...
            / sqrt(nmis);
         if useggm
              disp(sprintf('   \t%3i  \t %8.2e  \t%9.3e \t   %10.3e \t\t   %3.1f',it,peff_ave, dXmis, rdXmis,sp_level))
          else
              disp(sprintf('   \t%3i  \t %8.2e  \t%9.3e \t   %10.3e',it, peff_ave, dXmis, rdXmis))
          end
         
      else
          if useggm
              disp(sprintf('   \t%3i  \t %8.2e  \t%9.3e \t   %10.3e \t\t   %3.1f',it,peff_ave, dXmis, rdXmis,sp_level))
          else
              disp(sprintf('   \t%3i  \t %8.2e  \t%9.3e \t   %10.3e',it, peff_ave, dXmis, rdXmis))
          end
      end
   end
end                                       % EM iteration

% add mean to centered data matrix
X  = X + repmat(M, n, 1);

return   
   
