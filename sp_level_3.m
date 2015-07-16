function [level_TT, level_TP, level_PP, adj] = sp_level_3(O, ind_T, ind_P)

    p = size(O,1);
    d = diag(diag(O));
    O_scaled = inv(sqrt(d))*O*inv(sqrt(d));
    adj = abs(O_scaled) > 1e-3;
    adj = adj - diag(diag(adj)); 
    
    p_TT = length(ind_T);
    p_PP = length(ind_P);
    
    adj_TT = adj(ind_T, ind_T);
    level_TT = sum(sum(adj_TT)) / (p_TT*(p_TT-1))*100;
    
    adj_TP = adj(ind_T, ind_P);
    level_TP = sum(sum(adj_TP)) / (p_TT*p_PP)*100;
    
    adj_PP = adj(ind_P, ind_P);
    level_PP = sum(sum(adj_PP)) / (p_PP*(p_PP-1))*100;
    
    adj = adj + eye(p);

end