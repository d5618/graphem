function C = matern_cov(d,scale,sigma,nu);
% function C = matern_cov(d,scale,sigma,nu);
%   Generate a Matern covariance matrix from vector of distances in d,
%   with parameters scale, sigma and nu, following the formulation:
%  C(d) = sigma^2/(2^(nu-1))/gamma(nu)*(2*sqrt(nu)*d/scale)^nu * besselk(nu,2*sqrt(nu)*d/scale);
%  
%  Julien Emile-Geay, USC, 2011
% ====================================================================
R = matern(scale,sigma,nu,d);
C = toeplitz(R);
return

function f = matern(scale,sigma,nu,d)
pos = find(d>eps);
dp = d(pos); % positive distance
f(pos) = sigma^2/(2^(nu-1))/gamma(nu) * (2*sqrt(nu)*dp/scale).^nu .* besselk(nu,2*sqrt(nu)*dp/scale);
f(d<=eps) = sigma^2;  % treat singularity at d = 0

if ~isempty(find(d < 0))
    error('All distances must be greater than or equal to zero')
end
return
    
    
