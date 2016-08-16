addpath('../');

% Generate data
rng(1)
% Temperature
field = rand(10,5);   % 10 years, 5 grid points
field(5:10,:) = nan;  % Insert missing values
field(1,2) = nan;     % We can also have missing values in the instrumental period.

% Proxies
proxy = rand(10,3);   % 10 years, 3 proxies

% calibration period
calib = 1:4;

% Reconstruct field using default parameters (glasso)
opt.regress = 'ols';
opt.useggm = 1;
opt.method = 'glasso';
opt.weights = ones(5,1);  % Weights to compute the space average
opt.err_export = 0;
opt.target_sparsity = {3,3,3};

[field_r, diagn] = graphem_cfr(field,proxy,calib, opt);
