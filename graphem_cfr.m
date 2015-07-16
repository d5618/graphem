function [field_r,diagn] = graphem_cfr(field,proxy,calib,opt)
% Function [field_r,diagn] = graphem_cfr(field,proxy,calib,opt)
%  Performs graphem-based climate reconstruction
%
%  inputs:  - field, 2D climate field (nf x pf)
%           - proxy, proxy matrix (np x pp)
%           - calib, index of calibration set (< np)
%           - options, structure of optional GraphEM parameters.
%
%    The graph is specified through opt.method, which can take the
%      following values:
%    - 'graph_given': the graph is externeally specified in which
%       case an adjacency matrix must be provided through opt.adj.
%
%    -  'glasso':   Graphical Lasso (L1-penalized model selection)
%      the graph is estimated over the calibration interval using fastpath.m (if opt.target_sparsity
%      is a singleton) or greedy_search (if opt.target_sparsity is a cell triplet {TT,TP, PP},
%      describing the targeted sparsity in the corresponding regions of the adjacency matrix)
%      Since graph estimation requires a complete data matrix, it is infilled using RegEM
%      with individual ridge regression ('iridge') if it was incomplete.
%       Default opt.target_sparsity = 5 (always expressed in %).
%
%    - 'neigh_graph': a neighborhood graph is created, gathering all points
%    within opt.cutoff_radius (in km) of each original point on the grid.
%
%    - 'neigh_graph_ind_PP': same as 'neigh_graph' but the proxies are
%       conditionally indepednent given the temperature field, so the PP part
%       of the adjacency atrix is diagonal.
%     - 'neigh_TT_CAR_TP_ind_PP' same as 'neigh_graph_ind_PP', but the
%        proxy-temperature relations edges limited to the k nearest neighbors
%         (N is specified via opt.n_neighbors. default = 10)
%

%    - 'neigh_graph_CAR_ind_PP', same as 'neigh_graph_ind_PP, but both the
%        proxy-temperature and temperature-tempearture edges are limited to
%        the N nearest neighbors
%         (N is specified via opt.n_neighbors. default = 10)
%
%   <<< TO DO:  DESCRIBE ALL OPTIONS AND THEIR USE !! >>>
%
%
%   Most of these options require
%
%  outputs: - field_r, reconstructed field (np x pf)
%           - diagn, structure of diagnostic outputs, including
%              * Xerr, estimate of imputation error
%              * avail, vector of available values
%              * miss, vector of missing values
%              * iptrn, index of pattern to which each row belongs
%              * B, regression matrix
%              * RE,R2: "reduction of error" and R-squared in-sample statistics
%                  (computed iff options.insample = 1).
%              * peff, number of effective parameters
%
% All outputs are given for each pattern of missing values, except the last
% three: RE and R2 have dimensions (np x pf), and peff is a np x 1 vector.
%
% History: created  15-Aug-2014 14:47   by Julien Emile-Geay (USC) based
%           eponymous code by D.Guillot (Stanford)
%
% ===========================================================================

% Process options
% ===============
if nargin < 3 || isempty(opt)
    fopts = [];
    opt.regress = 'ols';
    opt.useggm = 1;
    opt.method = 'glasso';
else
    fopts = fieldnames(opt);
end

if strmatch('insample', fopts)  % compute in-sample skill or not?
    insample = opt.insample;
else
    insample = 0;
end

if strmatch('temp0', fopts)  % initial guess for temperature
    guess = 1;
    temp0 = opt.temp0;
else
    guess = 0;
end

% assign regularization method
if ~strmatch('regress', fopts)
    opt.regress = 'ols';
    opt.useggm = 1;
end

if ~isfield(opt,'useggm')
    opt.useggm = 1;
end

if isfield(opt,'n_neighbors')
    n_neighbors = opt.n_neighbors;
else
    n_neighbors = 10;
end


if ~isfield(opt,'method')
    error('graphem_cfr: you must specify a method to choose the graph\nAvailable options are: `cte_glasso` and `cte_adj`');
end

if isfield(opt, 'filter_graph')
    if opt.filter_graph == 1 & ~isfield(opt,'lonlat')
        error('graphem_cfr: You need to specify the opt.lonlat variable amd a cutoff radius to trim the graph');
    end
    
    if opt.filter_graph == 1 & ~isfield(opt, 'cutoff_radius')
        opt.cutoff_radius = 4000;  % Use a 4000 km radius if the radius has not been provided
    end
else
    opt.filter_graph = 0;
end

if isfield(opt, 'filter_tempgraph')
    if opt.filter_tempgraph == 1 & ~isfield(opt,'lonlat')
        error('graphem_cfr: You need to specify the opt.lonlat variable to trim the graph (temp points only)');
    end
    
    if opt.filter_tempgraph == 1 & ~isfield(opt, 'cutoff_radius')
        opt.cutoff_radius = 4000;  % Use a 4000 km radius if the radius has not been provided
    end
else
    opt.filter_tempgraph = 0;
end

if ~isfield(opt, 'target_sparsity')
    opt.target_sparsity = 5;  % uniform sparsity of 5%
end

if ~isfield(opt,'nsample')
    opt.nsample = 100;
end

if ~isfield(opt,'err_export')
    opt.err_export = 1;
end

if isfield(opt, 'screen_proxies')
    if opt.screen_proxies == 1
        if ~isfield(opt, 'screen_radius')
            opt.screen_radius = 3000;  % Default 3000 km radius
        end
        
        if ~isfield(opt, 'screen_corr')
            opt.screen_corr = 0.3;   % Keep proxy if it has a neighbor with 0.3 correlation by default
        end
    end
else
    opt.screen_proxies = 0;
end

%   ====== end options processsing  =======

[nf,pf]             = size(field); % field dimensions
[np,pp]             = size(proxy); % proxy matrix dimensions

%	Assemble climate field/proxy data matrices
X_in                = NaN(np,pf+pp);
inst                = field(calib,:); % instrumental field
X_in(calib,1:pf)    = inst;   % Put in instrumental data
X_in(:,pf+1:pf+pp)  = proxy;  % add (possibly incomplete) proxy data

%  Initial Guess
%==================================
if guess
    X0 =   NaN(np,pf+pp);
    X0(:,1:pf)       = temp0;  % Put in temperature data
    X0(:,pf+1:pf+pp) = proxy;  % the rest is (possibly incomplete) proxy data
    opt.X0 = X0;
    opt.C0 = cov(X0);   % very rough first guess ; needs refinement
    opt.M0 = nmean(X0);   % very rough first guess ; needs refinement
end

if isfield(opt,'adj')
    opt.method = 'given_graph';
end

switch opt.method
    case 'given_graph'
        if ~isfield(opt,'adj')
            error('graphem_cfr: graph not provided')
        end
        disp('Using specified graph at every iteration')
        
    case 'glasso'
        disp('Graph chosen using the graphical lasso')
        if ~isfield(opt,'nmax')
            opt.nmax = 30;
        end
        % Restrict field to the calibration period to estimate the
        % graph
        Xcal = X_in(calib,:);
        
        % Reconstruct missing values over the calibration period if necessary
        
        if any(any(isnan(Xcal)))
            fprintf('Warning: missing values over the calibration period will be reconstructed using RegEM iRidge\n')
            opt0.regress='iridge';
            opt0.useggm = 0;
            opt0.maxit = 200;
            opt0.stagtol = 5e-3;
            Xcal = graphem(Xcal, opt0);
        end
        
        if opt.screen_proxies == 1
            fprintf('\nScreening proxies to have a correlation of at least %1.4f within a radius of %1.4f\n', opt.screen_corr, opt.screen_radius)
            [ind_screen, screen_pp] = screen_proxies(Xcal, pf, pp, opt.screen_radius, opt.screen_corr, opt.lonlat);
            fprintf('%d of the %d proxies have been removed\n', pp-screen_pp, pp)
            pp = screen_pp;
            Xcal = Xcal(:,ind_screen);
            X_in = X_in(:,ind_screen);
        end
        
        % Compute path of adajcency matrices
        
        fprintf('Estimating the graph using the calibration period\n\n')
        if numel(opt.target_sparsity) == 1
            [adjmat, sp] = fastpath(Xcal,opt.target_sparsity,opt.nmax);
            nb_models = size(sp,1);
            opt.adj = adjmat{nb_models-1};
            fprintf('\nTargeted sparsity level: %1.4f\n', opt.target_sparsity)
            fprintf('Selected model sparsity level: %1.4f\n\n', sp(nb_models-1))
        elseif numel(opt.target_sparsity) == 3
            [TT,TP,PP] = deal(opt.target_sparsity{:});
            % Find adjacency matrix using greey algorithm
            [sp, adjmat] = greedy_search(corrcoef(Xcal), 1:pf, (pf+1):(pf+pp), TT, TP, PP, opt.nmax);
            nb_models = size(sp,1);
            opt.adj = adjmat{nb_models-1}; % pick second to last graph (just before going over the edge)
        elseif numel(opt.target_sparsity) == 2
            [TT,TP] = deal(opt.target_sparsity{:});
            % Find adjacency matrix using greey algorithm
            [sp, adjmat] = greedy_search_d(corrcoef(Xcal), 1:pf, (pf+1):(pf+pp), TT, TP, opt.nmax);
            nb_models = size(sp,1);
            opt.adj = adjmat{nb_models-1}; % pick second to last graph (just before going over the edge)
        end
        
        % Filter the graph if desired
        if opt.filter_graph == 1
            fprintf('Trimming the total graph using a cutoff radius of %1.2f km\n', opt.cutoff_radius)
            opt.adj = filter_adj(opt.adj, opt.lonlat, opt.cutoff_radius);
            % Update sparsity level
            adj_tmp = opt.adj - diag(diag(opt.adj));
            sparsity_adj = sum(sum(adj_tmp))/((pp+pf)*(pp+pf-1))*100;
            fprintf('Filtered model sparsity level: %1.4f\n\n', sparsity_adj)
        end
        if opt.filter_tempgraph == 1
            fprintf('Trimming the temperature graph using a cutoff radius of %1.2f km\n', opt.cutoff_radius)
            adj_f = filter_adj(opt.adj(1:pf,1:pf),opt.lonlat, opt.cutoff_radius);
            opt.adj(1:pf,1:pf) = adj_f;
            % Update sparsity level
            adj_tmp = opt.adj - diag(diag(opt.adj));
            sparsity_adj = sum(sum(adj_tmp))/((pp+pf)*(pp+pf-1))*100;
            fprintf('Filtered model sparsity level: %1.4f\n\n', sparsity_adj)
        end
        
    case {'neigh_graph', 'neigh_graph_ind_PP', 'neigh_graph_CAR_ind_PP', 'neigh_TT_CAR_TP_ind_PP'}  % Use a nearest-neighbors graph
        disp('Using neighborhood graph')
        if ~isfield(opt,'lonlat')
            error('graphem_cfr: you must specify the coordinates of the locations to use a nearest-neighbors graph');
        end
        
        if ~isfield(opt,'cutoff_radius')
            error('graphem_cfr: you must specify a cutoff radius to use a nearest-neighbors graph');
        end
        
        opt.adj = neigh_radius_adj(opt.lonlat, opt.cutoff_radius);
        adj_CAR = neigh_size_adj(opt.lonlat,n_neighbors); % select the n_neighbors closest neighbors
        
        if strcmp(opt.method,'neigh_graph_ind_PP') | strcmp(opt.method,'neigh_graph_CAR_ind_PP') | strcmp(opt.method,'neigh_TT_CAR_TP_ind_PP')
            opt.adj((pf+1):(pf+pp), (pf+1):(pf+pp)) = eye(pp);
        end
        
        if strcmp(opt.method,'neigh_graph_CAR_ind_PP')
            opt.adj(1:pf,1:pf) = adj_CAR(1:pf,1:pf);
        end
        
        if strcmp(opt.method,'neigh_TT_CAR_TP_ind_PP')
            opt.adj((pf+1):(pf+pp),1:pf) = adj_CAR((pf+1):(pf+pp),1:pf);
            opt.adj(1:pf,(pf+1):(pf+pp)) = adj_CAR(1:pf,(pf+1):(pf+pp));
        end
        
        % Compute model sparsity
        adj_tmp = opt.adj - diag(diag(opt.adj));
        sparsity_adj = sum(sum(adj_tmp))/((pp+pf)*(pp+pf-1))*100;
        fprintf('Nearest-neighbors model sparsity level: %1.4f\n\n', sparsity_adj)
        
    case {'hybrid'}
        disp('Use a neighbor-glasso hybrid graph')
        if ~isfield(opt,'nmax')
            opt.nmax = 30;
        end
        % Restrict field to the instrumental temperature to estimate the graph
        Xcal = inst;
        
        % Reconstruct missing values over the calibration period if necessary
        if any(any(isnan(Xcal)))
            fprintf('Warning: missing values over the calibration period will be reconstructed using RegEM iRidge\n')
            opt0.regress='iridge';
            opt0.useggm = 0;
            opt0.maxit = 200;
            opt0.stagtol = 5e-3;
            Xcal = graphem(Xcal, opt0);
        end
        
        % TT: Glasso graph
        disp('1) Find a glasso graph for TT')
        TT = deal(opt.target_sparsity{1});
        [sp,adjmat] = greedy_search_TT(corrcoef(Xcal),TT,opt.nmax);
        nb_models = size(sp,1);
        opt.adj(1:pf,1:pf) = adjmat{nb_models-1};
        
        % TP: Neighborhood graph
        % disp(['2) Find a neighborhood graph for TP, with a cutoff radius: ' num2str(opt.cutoff_radius) 'km'])
        % adj_CAR = neigh_radius_adj(opt.lonlat, opt.cutoff_radius);
        disp(['2) Find a CAR neighborhood graph for TP'])
        adj_CAR = neigh_size_adj(opt.lonlat,n_neighbors); % select the n_neighbors closest neighbors
        opt.adj((pf+1):(pf+pp),1:pf) = adj_CAR((pf+1):(pf+pp),1:pf);
        opt.adj(1:pf,(pf+1):(pf+pp)) = adj_CAR(1:pf,(pf+1):(pf+pp));
        
        % PP: Diagonal
        disp('3) Set the PP part to be diagonal')
        opt.adj((pf+1):(pf+pp), (pf+1):(pf+pp)) = eye(pp);
        
        % Compute model sparsity
        adj_tmp = opt.adj - diag(diag(opt.adj));
        sparsity_adj = sum(sum(adj_tmp))/((pp+pf)*(pp+pf-1))*100;
        fprintf('Hybrid model sparsity level: %1.4f\n\n', sparsity_adj)
        
    case {'cor_graph'}
        disp('Use a correlation graph')
        % Restrict field to the instrumental temperature to estimate the graph
        Xcal = X_in(calib,:);
        
        % Reconstruct missing values over the calibration period if necessary
        if any(any(isnan(Xcal)))
            fprintf('Warning: missing values over the calibration period will be reconstructed using RegEM iRidge\n')
            opt0.regress='iridge';
            opt0.useggm = 0;
            opt0.maxit = 200;
            opt0.stagtol = 5e-3;
            Xcal = graphem(Xcal, opt0);
        end
        
        % TT & TP: correlation graph
        disp('1) Find a correlation graph for TT & TP, PP is diagonal')
        opt.adj = threshpath(corrcoef(Xcal),1:pf,(pf+1):(pf+pp),opt.levels);
                
        % Compute model sparsity
        adj_tmp = opt.adj - diag(diag(opt.adj));
        sparsity_adj = sum(sum(adj_tmp))/((pp+pf)*(pp+pf-1))*100;
        fprintf('Hybrid model sparsity level: %1.4f\n\n', sparsity_adj)
        
        
    otherwise
        error('graphem_cfr: Unknown method');
end
fig('Estimated Graph')
spy(opt.adj);
title('Graph choice','FontName','Times','FontSize',14,'FontWeight','bold')
export_fig(['./figures/' opt.tag '_graph.pdf'],'-r300','-cmyk');
% apply GraphEM
fprintf('Performing reconstruction\n')
[X, M, C, B, S, avail, miss, iptrn, peff] = graphem(X_in,opt);

% extract reconstructed field
field_r = X(:,1:pf);

% compute predicted values of the field over the calibration interval
% Xp = zeros(size(field(calib,:))); %output array
nc   = size(inst,1);
npat = length(avail); % number of patterns
  
% Resample from covariance residual if uncertainty band is exported
if opt.err_export  
	disp('Exporting an error term of the reconstruction')
	Xerr = zeros(np,pf+pp,opt.nsample);
	for j = 1:npat-1
   	 % display(['pattern ',num2str(j)])
   	 prow = find(iptrn ==j); mp = length(prow);
   	 pm   = length(miss{j});
   	 S{j} = (S{j}+S{j}.')/2; 
   	 for k = 1:opt.nsample
      	  Xerr(prow,miss{j},k) = mvnrnd(zeros(1,pm),S{j},mp);
   	 end
	end
end

% remove mean
X = X - repmat(M, [np 1]);

j = npat - 1; % no need to compute predictions using all patterns of missing values
if sum(isnan(B{j}))==0
    % make instrumental prediction
    X_hat = X(calib, avail{j}) * B{j};
    Xp    = X_hat(:,1:pf);
else
    warning('Prediction over the calibration period could not be performed')
    warning('Check that proxy availability does not vary at t = min(ti)')
    Xp    = X(:,1:pf);
end
% for j = 1:npat
%     if ~isempty(B{j}) && sum(ismember(1:pf,miss{j}))==pf
%        % make instrumental prediction
%        X_hat{j} = X(calib, avail{j}) * B{j};
%     end
% end
% Xp = X_hat{end}(:,1:pf);

% add mean to centered data matrices
Xp = Xp + repmat(M(1:pf), [nc 1]);
% assign predicted values of the field over the calibration interval
field_r(calib,:) = Xp(:,1:pf);

% give peff a more user-friendly form
peff_e = zeros(np,1);
for k = 1:npat
    if ~isempty(peff{k})
        peff_e(iptrn==k) = mean(peff{k});
    else
        peff_e(iptrn==k) = mean(peff{j}); % assign peff from the penultimate pattern of missing values
    end
end

% Uncertainty estimation over the instrumental period
if opt.err_export
	 covRes = cov(field_r(calib,:) - field(calib,:));
	 covRes(isnan(covRes)) = 0;
	 for k = 1:opt.nsample
   	  covRes = (covRes + covRes')/2;
   	  yerr(:,:,k)   = mvnrnd(zeros(1,pf),covRes,nc);
	 end
	 Xerr(calib,1:pf,:) = yerr;
    diagn.err   = Xerr(:,1:pf,:);
end

% Assign output data structures
diagn.avail = avail;
diagn.miss  = miss;
diagn.peff  = peff_e;
diagn.iptrn = iptrn;
diagn.B     = B;
diagn.adj   = opt.adj;

if insample
    [Xp,RE,R2] = insample_pred_regem(X,M,B,[1:pf],[pf+1:pp],calib,avail,iptrn);
    diagn.RE = RE;
    diagn.R2 = R2;
end

return

end


