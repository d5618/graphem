function [adjmat, sparsity] = fastpath(X,t_sparsity, nmax)

    S = corrcoef(X);
    p = size(S,1);
    lmax = max(max(S-diag(diag(S))));
    lmin = 0.05*lmax;
    lambdas = fliplr(linspace(lmin,lmax,nmax));
    i = 1;
    
    sparsity(i) = 0;
    adjmat{i} = eye(p);
    W = zeros(p,p);
    O = zeros(p,p);
    
    while i < nmax & sparsity(i) < t_sparsity
        i = i+1;
        [W,O] = graphical_lasso(S,lambdas(i));
        [sparsity(i), adjmat{i}] = sp_level(O);
    end

end
