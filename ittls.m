function [B, S, trunc, peff] = ittls_sqrtm(X,C, colX, colY, dof, kvec, quant_min);
%ITTLS  Individual TTLS regressions with Bayesian Model Averaging.
%      
%       Based on Cholesky factorizations and SVD only, hence eschews
%         eigendecompositions of covariance matrix
%
%   [B, S, trunc, peff] = ITTLS(X, C, colX, colY, dof, kvec, quant_min); 
%   returns a regularized estimate B of the coefficient matrix for the 
%   multivariate multiple regression model Y = X*B + noise(S).  
% 	     
%		Each column B(:,k) of B is computed by a TTLS regression as
%					 B(:,k) = Mxx_k Cxy(:,k), 
%	 where Mxx_k is a regularized inverse of Cxx in TTLS sense. 
%
%             
%   Given matrices X and Y, the total least squares (TLS) problem
%    consists of finding a matrix Br that satisfies
%
%                (X+dX)*B = Y+dY.
%
%    The solution must be such that the perturbation matrices dA
%    and dX have minimum Frobenius norm rho=norm( [dX dY], 'fro')
%    and each column of B+dB is in the range of X+dX [1].
%  
%    ITTLS computes the minimum-norm solution Br of the TLS problem,
%      truncated at ranks r in [Rmin,Rmax] [2]. 
%		The solution Br of this truncated TLS problem is a
%    regularized error-in-variables estimate of regression
%    coefficients in the regression model Y = X*B + noise(S). The
%    model may have multiple right-hand sides, represented as columns
%    of B.
%	
%    The optimal B matrix is determined by Bayesian Model Averaging of  all 
%    TTLS solutions in the specified range kvec = [kmin:kmax]. Comoutation of
%    model weigths,which involves the computation of the filter factors [2, 4],
%    also used for assembling the residual covariance matrix S.
%    
%
%    As input, ITTLS requires the data maxtrix X, the Covariance matrix C,
%	  the column index of available values (colX), the column index of 
%	  missing values (colY) and the truncation vector kvec.
%
%
%    ITTLS returns :
%		-  the rank-ro truncated TLS solution Br
%		- the matrix Sr = dY'*dY, which is proportional to the estimated covariance
%    matrix of the residual dY.  
%		- Vectors of optimal truncation levels 'trunc', and the corresponding
%		number of effective parameters 'peff'.
%
%    Optional arguments: 
%     -  quant_min : minimum quantile of BMA weights to define Occam's
%       window. Default = 0.1
%
%     References: 
%     [1] Van Huffel, S. and J.Vandewalle, 1991:
%         The Total Least Squares Problem: Computational Aspects
%         and Analysis. Frontiers in Mathematics Series, vol. 9. SIAM.
%     [2] Fierro, R. D., G. H. Golub, P. C. Hansen and D. P. O'Leary, 1997:
%         Regularization by truncated total least squares, SIAM
%         J. Sci. Comput., 18, 1223-1241
%     [3] Golub, G. H, and C. F. van Loan, 1989: Matrix
%         Computations, 2d ed., Johns Hopkins University Press,
%         chapter 12.3 
%		[4] Sima, D.M. and  Van Huffel, S. 2007 :
%			 "Level choice in truncated total least squares",
%	 		Comput. Stat. & Data Analysis, 52, 1104-1118, 
%					doi:10.1016/j.csda.2007.05.015
%		[5]  Schneider, T., 2001: Analysis of Incomplete Climate Data: 
%			Estimation of Mean Values and Covariance Matrices and Imputation 
%				of Missing Values. J. Clim., 14, 853--871. 

%
%    History : created Jan 28, 2010
% 
%      by Julien Emile-Geay (University of SOuthern California)
%           Diana Sima  (Katholieke Universiteit Leuven)
%		      Tapio Schneider (CalTech)
%			
%	 (Some Rights Rserved) 	Hepta Technologies, 2010.
%  =========================================================================
  %error(nargchk(6, 8, nargin))     % check number of input arguments 
  px  		= length(colX);
  py  		= length(colY);
  neigs   	= min(dof,px);  % rank of covariance matrix Cxx (=Sigma_aa)
  
  % ==============        truncation options        ========================
  
  truncmax = min(px-1,neigs);  % Define maximum truncation
  
  if nargin < 6 
     kvec   = [1:truncmax];
  elseif (nargin >= 6 &  max(kvec)>=truncmax & truncmax > min(kvec));  
     kvec=[min(kvec):truncmax];
  end
  kl=length(kvec);

  % ==============           process options  ========================
  if nargin < 7 
     quant_min = 0.1;
  end
  
  % =================           end options        =========================

  if nargout > 1
    S_out      = 1==1;
  else
    S_out      = 0==1;
  end
  
  % Define partition covariance matrix 		
  Cxy		   = C(colX,colY);
  Cyy          = C(colY,colY);
  Cxx		   = C(colX,colX);
  
  Xa           = X(:,colX)/sqrt(dof);  % might not work as X does not know about C
  Xm           = X(:,colY)/sqrt(dof);
  [m, n]       = size(Xa);
  
  % Find square-root factorization and singular values   
  [Vp,d]        = eig(Cxx);
  Sp            = diag(sqrt(d));
  Ra            = Vp*diag(Sp)*Vp';  
  Si            = 1./Sp; % pseudoinverse of Sp
  Si(Sp<sqrt(eps)) = 0;  
  
  % older solutions
  %Ra = cholinc(sparse(Cxx),1e-6); + csvd
  %[R, alpha, cond] = sqrtm(Cxx);
   
  % Fourier coefficients. (The following expression for the Fourier
  % coefficients is only correct if Cxx = X'*X and Cxy = X'*Y for
  % some, possibly scaled and augmented, data matrices X and Y; for
  % general Cxx and Cxy, all eigenvectors V of Cxx must be included,
  % not just those belonging to nonzero eigenvalues.)
   
  F            =  diag(Si) * Vp' * Cxy; % Eq(17) in [5]	
  
  % Part of residual covariance matrix that does not depend on the
  % regularization parameter:
  if (dof > px) 
    S0         = Cyy - F'*F;
  else
    S0         = sparse(py, py);
  end   
  
  % initialize outputs
  trunc        = zeros(py, 1);    % BM averaged truncation rank
  peff         = zeros(py, 1);    % effective number of parameters 
  B            = zeros(px, py);   % BM averaged regression matrix
  f            = zeros(px, py);   % BM averaged filter factors
  
  if S_out
    S          = zeros(py, py);
  end

  % Loop over missing values
  % compute kl TTLS solutions (one for each truncation level) and
     % associated filter factors
  
  parfor i = 1:py  
     % Vector of missing values 
     b         = Xm(:,i);   
     % Compute Cholesky update of Xa by adding column Xm(;,i);         
     R1 = cholinsert(Ra, b, Xa);
      
     % Compute SVD of augmented data matrix
        % (The singular values and right singular vectors of R1, defined 
        % this way are the same as those of [Xa b])  
     [U1,S1,V1] = csvd(R1); % compact SVD
     
     % Solve TTLS problem and compute filter factors
	  [Bi,ff,bic,peff_i,weight] = ttls_ff(Sp,Si,S1,V1,kvec,Xa,b);   
    
     % Define Occam Window
     %occam = kvec; 
     occam = find(weight >= quantile(weight,quant_min));
     % Renormalization
     weight    = weight(occam)/sum(weight(occam));
     
    % Assign matrix of regression coefficients, filter factors, peff 
    %   by taking expectance over posterior model probabilities
     B(:  ,i)   = Bi(:,occam) * weight; 
     trunc(i)   = kvec(occam) * weight;
     peff( i)   = peff_i(occam) * weight;
     f(:, i)    = ff(:,occam) * weight;     
  end
  
  
  % assemble estimate of covariance matrix of residuals	
  for i = 1:py	
     for j = 1:i
        diagS   = Sp.*(1-f(:,j)).*(1-f(:,i)).*Si;
        S(j,i)  = S0(j,i) + F(:,j)' * (diagS .* F(:,i));
        S(i,j)  = S(j,i); % symmetrize
     end   
  end
  
return
    
% Fast Cholesky insert and remove functions
% Updates R in a Cholesky factorization R'R = X'X of a data matrix X. R is
% the current R matrix to be updated. x is a column vector representing the
% variable to be added and X is the data matrix containing the currently
% active variables (not including x).
% 
% Author: Karl Skoglund, IMM, DTU, kas@imm.dtu.dk
 
function R = cholinsert(R, x, X)
diag_k = x'*x; % diagonal element k in X'X matrix
if isempty(R)
  R = sqrt(diag_k);
else
  col_k = x'*X; % elements of column k in X'X matrix
  R_k = R'\col_k'; % R'R_k = (X'X)_k, solve for R_k
  R_kk = sqrt(diag_k - R_k'*R_k); % norm(x'x) = norm(R'*R), find last element by exclusion
  R = [R R_k; [zeros(1,size(R,2)) R_kk]]; % update R
end

