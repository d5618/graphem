% greedy_search.m
% Dominique Guillot (dguillot@stanford.edu)
% 2012
%
% Input: - S = sample covariance matrix of the field
%        - ind_T = temperature indices
%        - ind_P = proxy indices
%        - target_TT, target_TP, target_PP = target sparsity for the
%             different parts of the precision matrix
%        - N = maximum number of regularization parameters
%        

function [sparsity, adjmat] = greedy_search(S, ind_T, ind_P, target_TT, target_TP, target_PP, N)

    p = size(S,1);
    pen = zeros(p,p);
    rho_max = max(max(S-diag(diag(S))));
    rho_min = 0.1*rho_max;
    
    O = eye(p);
    W = eye(p);
    
    rhos = linspace(rho_min, rho_max, N);
    
    c_TT = N;
    c_TP = N;
    c_PP = N;
    
    col = {'black', 'blue'};
    c_col = [1,1,1];
    
    iter = 1;
    stop = 0;
    
    visited = [];   % Visited positions
    
    fprintf('\nSearching for an optimal graph: \n\n')
    fprintf('Iter    TT      TP      PP\n')
    
    while stop == 0
        %fprintf('%1.3d  %1.2f  %1.2f  %1.2f\n', iter, rhos(c_TT),
        %rhos(c_TP), rhos(c_PP))
        
        % Build penalty matrix
        pen(ind_T,ind_T) = rhos(c_TT);
        pen(ind_T,ind_P) = rhos(c_TP);
        pen(ind_P,ind_T) = rhos(c_TP);
        pen(ind_P,ind_P) = rhos(c_PP);
        % Compute solution
        [O W opt cputime niter dGap] = QUIC('default', S, pen, 1e-3, 0, 1000, O, W);
        % Compute sparsity of the different parts
        [sparsity_TT(iter), sparsity_TP(iter), sparsity_PP(iter),adjmat{iter}] = sp_level_3(O, ind_T, ind_P);
        % Move to next point
        if(sparsity_TT(iter) < target_TT)
            c_TT = max(1,c_TT-1);
            c_col(1) = 1;
        else
            c_col(1) = 2;
        end
        
        if(sparsity_TP(iter) < target_TP)
            c_TP = max(1,c_TP-1);
            c_col(2) = 1;
        else
            c_col(2) = 2;
        end
        
        if(sparsity_PP(iter) < target_PP)
            c_PP = max(1,c_PP-1);
            c_col(3) = 1;
        else
            c_col(3) = 2;
        end
        
        fprintf('%1.3d  %6.3f  %6.3f  %6.3f\n', iter, sparsity_TT(iter), sparsity_TP(iter), sparsity_PP(iter))
%         cprintf('black','%1.3d  ', iter)
%         cprintf(col{c_col(1)}, '%6.3f  ', sparsity_TT(iter))
%         cprintf(col{c_col(2)}, '%6.3f  ', sparsity_TP(iter))
%         cprintf(col{c_col(3)}, '%6.3f  ', sparsity_PP(iter))
%         cprintf('black','\n')
        
        iter = iter + 1;
        
        c_point = c_TT-1 + (c_TP-1)*N + (c_PP-1)*N^2;
        
        
        if ismember(c_point, visited)
            stop = 1;
        else
            visited = [visited, c_point];
        end
    end
    
    sparsity = [sparsity_TT', sparsity_TP', sparsity_PP'];
    

end