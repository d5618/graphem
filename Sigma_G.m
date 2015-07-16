function [C,sp_level] = Sigma_G(C0,opt)
% Sigma_G  
%             [C,sp_level] = Sigma_G(C0,opt)
%     fits a graphical model to a sample covariance matrix C0    
%     Inputs: C0, options structure opt
%     Outputs: graphical estimate C, sparsity level
%
    p = size(C0,2);
    
    C = fitggm(C0, opt.adj, opt.ggm_tol, opt.ggm_maxit,opt.ggm_thre);
    
    % Compute sparsity level
    adj = opt.adj;
    adj1 = adj-diag(diag(adj));
    adj1 = triu(adj1);
    nb_edges = sum(sum(adj1));
    sp_level = 100*nb_edges /(p*(p-1)/2);

end