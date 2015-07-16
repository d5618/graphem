function [level,adj] = sp_level(O)

    p = size(O,1);
    d = diag(diag(O));
    O_scaled = inv(sqrt(d))*O*inv(sqrt(d));
    adj = abs(O_scaled) > 1e-3;
    adj = adj - diag(diag(adj));    
    level = sum(sum(adj))/(p*(p-1))*100;
    adj = adj + eye(p);

end