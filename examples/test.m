addpath('../');

% Generate data

% Temperature
rng(1)
field = rand(9,9)
field(5:9,1:3) = nan;

% Proxies
proxy = rand(9,3);


% calibration period
calib = 1:4;

% Reconstruct field using default parameters (glasso)
opt.regress = 'ols';
opt.useggm = 1;
opt.method = 'glasso';
opt.weights = ones(9,1);  % Weights to compute the space average
opt.err_export = 0;
opt.target_sparsity = {3,3,3};

[field_r, diagn] = graphem_cfr(field,proxy,calib, opt);

save('./test.mat')