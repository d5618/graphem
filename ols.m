function [B, S] = ols(Sigma,xind, yind)
% OLS  Ordinary least squares fit of the 
%      regression problem  Y = X'*B + \eps
%      given a (well-conditioned) covariance matrix
%  
%  inputs: - Sigma, covariance matrix
%          - xind, index of X columns
%          - yind, index of Y columns
% 

%S = inv(Omega(yind,yind));
%B = -1*Omega(xind,yind)*S;

B = inv(Sigma(xind,xind)) * Sigma(xind, yind);
S = Sigma(yind,yind) - Sigma(yind, xind) * B;

end