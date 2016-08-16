function [Sigma,Omega] = GraphicalLasso(S,lambda)

    p = size(S,1);
    Sigma = eye(p);
    Omega = eye(p);
    pen = lambda*ones(p,p);

    [Omega Sigma opt cputime niter dGap] = QUIC('default', S, pen, 1e-3, 0, 1000, Omega, Sigma);
end
