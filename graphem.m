function [X, M, C, kavlr, kmisr, iptrn, peff, D, Cf] = graphem(X, options)
% GraphEM  Imputation of missing values with regularized EM algorithm using a
%          graphical estimate of the covariance matrix
%
%    GraphEM replaces missing values
%    (NaNs) in the data matrix X [n x p] with imputed values forming np 
%    distinct patterns.  GraphEM returns:
%
%       X:     data matrix with imputed values substituted for NaNs,
%       M:     estimated mean of X, [p x 1]
%       C:     estimated covariance matrix of X, [p x p]
%       kavlr{j}: Indices of available values for each missingness pattern
%       kmisr{j}: Indices of missing values for each missingness pattern
%       iptrn(i): Index j of missingness pattern for row i of X
%       peff : Effective number of degrees of freedom for each missingness pattern
%       D:     scaling factor
%       Cf:    C from final iteration before scaling back to its original unit 
%
%       Note:  D and Cf are only needed in uncert.m to (re)calculate S  
%       
%    Missing values are imputed with an expectation maximization (EM) 
%    algorithm, where the covariance matrix is estimated using a graphical 
%    model. The graphical model provides a regularized version of the sample 
%    covariance matrix that is more precise and well-conditioned (assuming 
%    the estimated graphical structure of the field is sparse enough).
% 
%    In an iteration of the GraphEM algorithm, given estimates of the mean and of 
%    the covariance matrix are revised in three steps. First, for each record 
%    X(i,:) (row of X, i=1:n) with missing values, the regression parameters of 
%    the variables with missing values on the variables with available values are
%    computed from the estimates of the mean and of the covariance
%    matrix. Second, the missing values in a record X(i,:) are filled
%    in with their conditional expectation values given the available
%    values and the estimates of the mean and of the covariance
%    matrix, the conditional expectation values being the product of
%    the available values and the estimated regression
%    coefficients. Third, the mean and the covariance matrix are
%    re-estimated, the mean as the sample mean of the completed
%    dataset; and the covariance matrix as the maximum likelihood 
%    estimator of the covariance matrix under a normal model with the given 
%    graphical model structure and an estimate of the
%    conditional covariance matrix of the imputation error. As the
%    regression models need to be computed only for each unique pattern
%    of missing values in records (rows of X), the iterations are go over
%    unique missingness patterns. (This is considerably faster than
%    iterating over each row of X, e.g., when missingness has a staircase
%    pattern with few steps.)
%
%    As default initial condition for the imputation algorithm, the
%    mean of the data is computed from the available values, mean
%    values are filled in for missing values, and a covariance matrix
%    is estimated as the sample covariance matrix of the completed
%    dataset with mean values substituted for missing
%    values. Optionally, initial estimates for the missing values and
%    for the covariance matrix estimate can be given as input
%    arguments.
%
%    The GraphEM algorithm requires an approximation of the conditional 
%    independence structure of the field as an input. The 
%    conditional independence graph of the field must be provided 
%    by setting the variable options.adj to the adjacency matrix of the graph.
%
%
%    The OPTIONS structure specifies parameters in the algorithm:
%
%     Field name         Parameter                                  Default
%
%     OPTIONS.stagtol    Stagnation tolerance: quit when            5e-3
%                        consecutive iterates of the missing
%                        values are so close that
%                          norm( Xmis(it)-Xmis(it-1) )
%                             <= stagtol * norm( Xmis(it-1) )
%
%     OPTIONS.maxit      Maximum number of EM iterations.           50
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
%	  OPTIONS.M0		 Initial estimate of the mean vector.     not set
%
%
%     OPTIONS.Xcmp       Display the weighted rms difference        not set
%                        between the imputed values and the
%                        values given in Xcmp, a matrix of the
%                        same size as X but without missing
%                        values. By default, GRAPHEM displays
%                        the rms difference between the imputed
%                        values at consecutive iterations. The
%                        option of displaying the difference
%                        between the imputed values and reference
%                        values exists for testing purposes.
%
%    OPTIONS.use_iridge  Adds Ridge regularization to the estimated     0
%                        covariance matrix. (Used internally by 
%                        graphem_cfr.m to impute missing values in 
%                        instrumental field to estimate the graphical
%                        structure using the graphical lasso.)
%
%
%    References:
%    [1] Guillot, D., Rajaratnam, B., Emile-Geay, J., Statistical 
%        paleoclimate reconstructions via Markov random fields,
%        Ann. Appl. Stat. Volume 9, Number 1 (2015), 324-352., 
%        http://arxiv.org/abs/1309.6702.
%    [2] T. Schneider, 2001: Analysis of incomplete climate data:
%        Estimation of mean values and covariance matrices and
%        imputation of missing values. Journal of Climate, 14,
%        853--871. (RegEM algorithm)
%    [3] R. J. A. Little and D. B. Rubin, 1987: Statistical
%        Analysis with Missing Data. Wiley Series in Probability
%        and Mathematical Statistics. (For EM algorithm.)


if ndims(X) > 2  
	error('X must be vector or 2-D array.'); 
end
% if X is a vector, make sure it is a column vector (representing
% a single variable)
if length(X) == numel(X)
	X = X(:);
end

[n, p] = size(X);

% number of degrees of freedom for estimation of covariance matrix
dofC = n - 1;            % use degrees of freedom correction

% ==============           process options        ======================
if nargin ==1 || isempty(options)
	fopts = [];
else
	fopts = fieldnames(options);
end

% default values
regpar_given      = 0==1;
truncslct         = 'none';
trial_trunc_given = 0==1;


if strmatch('stagtol', fopts)
	stagtol = options.stagtol;
else
	stagtol = 5e-3;
end

if strmatch('maxit', fopts)
	maxit = options.maxit;
else
	maxit = 50;
end

if strmatch('inflation', fopts)
	inflation = options.inflation;
else
	inflation = 1;
end

if strmatch('Xmis0', fopts);
	Xmis0_given = 1==1;
	Xmis0       = options.Xmis0;
	if any(size(Xmis0) ~= [n,p])
		error('OPTIONS.Xmis0 must have the same size as X.')
	end
else
	Xmis0_given = 0==1;
end

if strmatch('C0', fopts);
	C0_given = 1==1;
	C0       = options.C0;
	if any(size(C0) ~= [p, p])
		error('OPTIONS.C0 has size incompatible with X.')
	end
else
	C0_given = 0==1;
end

if strmatch('M0', fopts);
	M0_given   = 1;
	M0         = options.M0;
else
	M0_given   = 0;
end

if strmatch('Xcmp', fopts);
	Xcmp_given = 1==1;
	Xcmp       = options.Xcmp;
	if any(size(Xcmp) ~= [n,p])
		error('OPTIONS.Xcmp must have the same size as X.')
	end
	sXcmp      = std(Xcmp);
else
	Xcmp_given = 0==1;
end

if strmatch('use_iridge', fopts);
	use_iridge = options.use_iridge;
else
	use_iridge = 0;
end

if strmatch('scalefac', fopts)
    D_given      = 1;
    D            = options.scalefac(:);
    if length(D) ~= p
        error('OPTIONS.scalefac must be vector of length p.')
    end
else
    D_given      = 0;
end

% Options for GGM

if strmatch('useggm',fopts);
	useggm = options.useggm;
	if useggm
		opt_ggm = options_ggm(options);
		Xerr = NaN;
    end
else
	useggm = 1;
end
% =================           end options        ========================


% get indices of missing values and initialize matrix of imputed values
indmis       = find(isnan(X));
nmis         = length(indmis);

if nmis == 0
	warning('No missing value flags found.')
	return                                      % no missing values. Exit.
end

[~,kmis]  = ind2sub([n, p], indmis);
% use full matrices: needs more memory but is faster
Xmis      = X;                            % matrix of imputed values
Xerr      = zeros(size(X));               % standard error imputed vals.

% find unique missingness patterns
[np, kavlr, kmisr, prows, mp, iptrn] = missingness_patterns(X);

disp(sprintf('\nGRAPHEM:'))
disp(sprintf('\tSample size:                     %4i', n))
disp(sprintf('\tUnique missingness patterns:     %4i', np))
disp(sprintf('\tPercentage of values missing:      %5.2f', nmis/(n*p)*100))
disp(sprintf('\tStagnation tolerance:              %9.2e', stagtol))
disp(sprintf('\tMaximum number of iterations:     %3i', maxit))

if (inflation ~= 1)
	disp(sprintf('\tResidual (co-)variance inflation:  %6.3f ', inflation))
end

if Xmis0_given && C0_given
	disp(sprintf('\tInitialization with given imputed values and covariance matrix.'))
elseif C0_given
	disp(sprintf('\tInitialization with given covariance matrix.'))
elseif Xmis0_given
	disp(sprintf('\tInitialization with given imputed values.'))
else
	disp(sprintf('\tInitialization of missing values by mean substitution.'))
end

if D_given
	disp(sprintf('\tUsing scaling for variables given by OPTIONS.scalefac.'))
else
	disp(sprintf('\tPerforming regularization on correlation matrices.'))
end

if useggm
	disp('Using graphical estimate of the covariance matrix')
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

% initial estimates of missing values
if Xmis0_given
    % substitute given guesses for missing values
	X(indmis)  = Xmis0(indmis);
	[X, M]     = center(X);        % center data to mean zero
else
	if M0_given
		M = M0;
		X = X - repmat(M0,n,1);
		X(indmis)  = zeros(nmis, 1);   % fill missing entries with zeros
	else
		[X, M]     = center(X);        % center data to mean zero
		X(indmis)  = zeros(nmis, 1);   % fill missing entries with zeros
    end
end

if C0_given
	C    = C0;
else
	C    = X'*X / dofC;      % initial estimate of covariance matrix
	C    = C + 0.1*eye(p);   % add small regularization for the first iteration
end


% initialize matrices needed for each missingness pattern
peff     = cell(np, 1);

it       = 0;
rdXmis   = Inf;

while (it < maxit && rdXmis > stagtol)
	it = it + 1;
    
    % initialize for this iteration ...
	CovRes = zeros(p, p);      % ... residual covariance matrix
	peff_ave = 0;                % ... average effective number of variables
    
    % scale variables
	if D_given
        % Strictly speaking, this would not need to be repeated at
        % each iteration when D is given, but easier to code that
        % way.
		[X, C]     = rescale(X, C, D);
	else
		[X, C, D]  = rescale(X, C); % scale to unit variance
	end
    
    
    % cycle over missingness patterns
	for j=1:np                     
		pm = length(kmisr{j}); % number of missing values in this pattern
        % disp(['pattern ',num2str(j)])
		if pm > 0
			pa = p - pm;           % number of available values in this pattern
            
			if pa < 1
				error('No available values in at least one row of the data matrix.')
			end
            
            % regression of missing variables on available variables
            
            % ordinary least squares regression without further
            % regularization (needs a sparse graph for a
            % well-conditioned C) - default.
            
            % Use iridge if specified (only used in graphem_cfr when estimating the graphical 
            % structure using glasso). 
            if use_iridge
				[B, S, ~, peff] = iridge(C(kavlr{j},kavlr{j}), C(kmisr{j},kmisr{j}), C(kavlr{j},kmisr{j}), n-1, []);
            else 
				[B, S]   = ols(C,kavlr{j},kmisr{j});
				Cf       = C;
				peff     = 0;
			end
            
            % missing value estimates
			Xmis(prows{j}, kmisr{j})  = X(prows{j}, kavlr{j}) * B;
            
            % inflation of residual covariance matrix
			S = inflation * S;
            
            % add up contributions to residual covariance matrix estimate in
            % original scaling
			CovRes(kmisr{j}, kmisr{j}) = CovRes(kmisr{j}, kmisr{j}) + mp(j)*S;
            
			peff_ave   = peff_ave + mp(j)*sum(peff)/nmis; % add up eff. number of variables
			dofS       = dofC - peff;
            

            % GCV estimate of standard error in imputed values
			Xerr(prows{j}, kmisr{j}) = repmat(( dofC * sqrt(diag(S)) ./ dofS)', mp(j), 1);
		end
	end     % end loop over patterns
    
    % rescale variables to original scaling
	X          = X .* repmat(D', n, 1);
	Xmis       = Xmis .* repmat(D', n, 1);
	C          = C .* repmat(D', p, 1) .* repmat(D, 1, p);
	CovRes     = CovRes .* repmat(D', p, 1) .* repmat(D, 1, p);
	 
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
        %  apply graphical structure to the sample covariance matrix
		[C,sp_level] = Sigma_G(cov(X)+CovRes/dofC,opt_ggm);
	else
		C = (X'*X + CovRes)/dofC;
	end
    
	if Xcmp_given
        % imputed values in original scaling
		Xmis(indmis) = X(indmis) + M(kmis)';
        % relative error of imputed values (relative to values in Xcmp)
		dXmis        = norm( (Xmis(indmis)-Xcmp(indmis))./sXcmp(kmis)' ) / sqrt(nmis);
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
end  % end of EM iteration

% add mean back to centered data matrix
X  = X + repmat(M, n, 1);