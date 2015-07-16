% greedy_search.m
% Dominique Guillot (dguillot@stanford.edu)
% 2012
%
% Input: - S = sample covariance matrix of the field
%        - ind_T = indices of the subset of the precision matrix
%        - target_TT = target sparsity for the subset of the precision matrix
%        - N = maximum number of regularization parameters
%        

function [sparsity, adjmat] = greedy_search_TT(S,target_TT, N)

    p = size(S,1);
    pen = zeros(p,p);
    rho_max = max(max(S-diag(diag(S))));
    rho_min = 0.1*rho_max;
    
    O = eye(p);
    W = eye(p);
    
    rhos = linspace(rho_min, rho_max, N);
    
    c_TT = N;

    col = {'black', 'blue'};
    c_col = 1;
    
    iter = 1;
    stop = 0;
    
    visited = [];   % Visited positions
    
    fprintf('\nSearching for an optimal graph: \n\n')
    fprintf('Iter    TT\n')
    
    while stop == 0
        % Build penalty matrix
        pen = rhos(c_TT);
        % Compute solution
        [O W opt cputime niter dGap] = QUIC('default', S, pen, 1e-3, 0, 1000, O, W);
        % Compute sparsity of the different parts
        [sparsity_TT(iter),adjmat{iter}] = sp_level(O);
        % Move to next point
        if(sparsity_TT(iter) < target_TT)
            c_TT = max(1,c_TT-1);
            c_col = 1;
        else
            c_col = 2;
        end
        
        fprintf('%1.3d  %6.3f\n', iter, sparsity_TT(iter))
        iter = iter + 1;
        
        c_point = c_TT-1;
        
        
        if ismember(c_point, visited)
            stop = 1;
        else
            visited = [visited, c_point];
        end
    end
    
    sparsity = sparsity_TT';
    

end
