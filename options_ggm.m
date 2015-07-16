function opt_ggm = options_ggm(options)

fopts = fieldnames(options);

if strmatch('adj',fopts);
    opt_ggm.adj = options.adj;
else
    error('Please provide a graph')
end

%     switch options.method
%       case 'cte_adj'
%         disp('Using specified graph at every iteration')
%         opt_ggm.adj = options.adj;
%       case {'neigh_graph', 'neigh_graph_ind_PP','neigh_graph_CAR_ind_PP','neigh_TT_CAR_TP_ind_PP'}
%         disp('Using neighborhood graph')
%         opt_ggm.adj = options.adj;
%       otherwise
%         error('options_ggm: No graph provided')
%     end

if strmatch('ggm_maxit',fopts);
    opt_ggm.ggm_maxit = options.ggm_maxit;
else
    opt_ggm.ggm_maxit = 200;
end

if strmatch('ggm_tol',fopts);
    opt_ggm.ggm_tol = options.ggm_tol;
else
    opt_ggm.ggm_tol = 5e-3;
end

if strmatch('ggm_thre',fopts);
    opt_ggm.ggm_thre = options.ggm_thre;
else
    opt_ggm.ggm_thre = 200;
end

end