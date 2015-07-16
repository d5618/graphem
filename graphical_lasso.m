function [Sigma,Omega] = GraphicalLasso(S,lambda,varargin)

    switch nargin
        case 2
            [Sigma,Omega] = huge_glasso(S,lambda); 
        case 3 
            [Sigma, Omega] = huge_glasso(S,lambda,varargin{1});
        case 4 
            [Sigma, Omega] = huge_glasso(S,lambda,varargin{1}, varargin{2});
    end

end