function [Xp,RE,R2] = insample_pred_regem(X,M,B,i_ind,p_ind,calib,avail,iptrn) 
% function [Xp,RE,R2] = insample_pred_regem(X,M,B,i_ind,p_ind,calib,avail,iptrn)
% 		
%    In-sample prediction of climate field from proxy records, using the new RegEM 
%    (as of Oct 2013). Note that the B matrices are the regression coeffients as 
%    of the last iteration of the algorithm, *before the final update of the 
%    mean and covariance matrices.* That is, they are the regression coeffients for 
%    the data centered with the penultimate estimate of the mean vector, not 
%    with the final estimate that is returned. With centered data as of the 
%    beginning of the last iteration, 
%
%    Xc = X - repmat(M-Mup, n, 1);
%
%    the regression coefficients give the imputed values in the original
%    scaling
%
%    Xc(prows{j}, kmisr{j}) = Xc(prows{j}, kavlr{j}) * B{j}
%
%    The difference should be very small if the algorithm has converged. 
%     
%
% INPUTS
%  - X,M,B,avail,iptrn: output of RegEM (cell arrays of dimension np, the
%  number of distinct patterns of missing values in X). 
%  - i_ind : index set of instrumental variables (e.g. temperature),
%  of length pi
%  - p_ind : index set of proxies 
%  - calib : index set of calibration period
%
%
% OUTPUTS:
%  - Xp: predicted X (ni x pi x np), one for each pattern
%  - RE, R2, statistics (n, pi), for a frozen network analysis. These are *in-sample*
%  statistics, so they overestimate the true skill (they are upper bounds
%  on such skill).  Their absolute value is thus irrelevant but they
%  illustrate the varying quality of the proxy-based prediction
%  as a function of the proxy network composition. 
% 
%
%   Note : CE = RE  since \mu_v = \mu_c;
%
%   CAUTION: assumes that time runs FORWARD, and that the proxy matrix is complete
%     over the instrumental period (easily relaxed if needed)
%
%  History: v0. 21-Nov-2013 15:24:06 Julien Emile-Geay, USC, 
% ======================================================================

np = length(avail);

[n,p] = size(X);
ni    = min(numel(calib),sum(calib));
pi    = length(i_ind);
Xc    = X(calib,i_ind);

% prepare output array
Xp = zeros(ni,pi,np);

% remove mean
X = X - repmat(M, n, 1);

for j=1:np             % cycle over patterns
   display(['Processing pattern ',int2str(j),' out of ',int2str(np)]);
   if sum(isnan(B{j}))==0
      % make instrumental prediction
      X_hat = X(calib, avail{j}) * B{j};
      Xp(:,:,j) = X_hat(:,i_ind);
   else % if this is the instrumental period, use the regression matrix over 
        % the previous pattern of missing values
      Xp(:,:,j) = Xp(:,:,j-1);
   end
end

% add mean to centered data matrices
Xp = Xp + repmat(M(i_ind), [ni 1 np]);

% compute verification statistics
pi = numel(i_ind); 
RE = NaN(n,pi); R2 = RE; CE = RE;

display('Computing validation statistics')
for j=1:np
   pattern = (iptrn==j); lp = sum(pattern);
   [REn,CEn,R2n]=verif_stats(Xc,squeeze(Xp(:,:,j)),1:ni,1:ni);
   RE(pattern,:) = repmat(REn,lp,1);
   CE(pattern,:) = repmat(CEn,lp,1);
   R2(pattern,:) = repmat(R2n,lp,1);
end


end