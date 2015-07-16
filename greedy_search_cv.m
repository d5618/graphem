function [optimal_sparsity, scores] = greedy_search_cv(temp_inst, proxy_inst, target_vals, N)
    
    nb_target = length(target_vals);
    scores = zeros(nb_target, 5);
    [n,pt] = size(temp_inst);
    % Builds 5-folds cross-validation indices
    indices = cv_indices(n,5);
    
    opt.useggm = 1;
    opt.regress = 'ggm';
    opt.method = 'cte_adj';
    opt.tol = 5e-3;
    opt.maxit = 100;
    opt.block = true;
    opt.nmax = N;
    opt.filter_graph = 0;
    opt.filter_tempgraph = 0;
    opt.ggm_l2reg = 0;

    for i1 = 1:nb_target
        opt.target_sparsity = {target_vals(i1),target_vals(i1),target_vals(i1)};
        for i2=1:5
            calib = indices ~= i2;
            verif = indices == i2;
            field_r = graphem_cfr(temp_inst,proxy_inst,1:n,calib,opt);
            scores(i1,i2) = mean(mean((field_r(verif,:) - temp_inst(verif,:)).^2));
        end
    end

    m_score = mean(scores,2);
    [m,best_score] = min(m_score);
    optimal_sparsity = target_vals(best_score);
end
