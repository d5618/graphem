% FITGGM:
% A modified regression algorithm for estimation of an undirected Gaussian
% graphical model with known structure
% Ref:  Hastie, Tibshirani, Friedman, 'The elements of statistical
%       learning', Algorithm 17.1, p. 634.
% Programmed by Dominique Guillot (dguillot@usc.edu)
% Created: March 1, 2011
% Last modified: 15-Aug-2014 by J.E.G (moved Tikhonov regularization 
%                              from main graphem routine to here)

% Input:   - S = sample covariance matrix of the data
%          - adj = the adjacency matrix of the graph representing the
%                  structure of the inverse covariance matrix
%          - tol = tolerance acceptable before terminating the algorithm
%            (default = 5e-3)
%          - maxit = maximum number of iterations  (default = 200)

function [W, Omega] = fitggm(S,adj, tol, maxit, thre)

% Set the default parameters 

if nargin == 2
    tol = 5e-3;
    maxit = 200;
    thre  = 50;
elseif  nargin == 3
    maxit = 200;
    thre  = 50;
elseif  nargin == 4
    thre  = 50;
end



p = size(S,1);   % Number of variables
pp = p*(p-1)/2;    

if cond(S) > thre  % if matrix is ill-conditioned (usually first few iterations)
    S = S + 0.1*eye(p); % apply small amount of Tikhonov regularization
end

W = S;   % Initial estimate of the covariance matrix



Omega = zeros(p,p);
beta = cell(1,p);

for iter = 1:maxit
    
    W_old = W;

    for j=1:p
         % Indices to be removed
         indices = adj(j,:) == 1;
         indices(j) = 0;   % We need to remove the j-th row and column
         W11_star = W(indices,indices);
         s12_star = S(indices,j);
         if ~all(indices==0)
             beta_star = W11_star\s12_star;   % Solve for beta^*
             %beta_star = precond(W11_star,s12_star);
             %beta_star = r_inv(W11_star,1e5)*s12_star;
             beta{j} = zeros(p,1);    % beta_star is padded with zeros to obtain beta
             beta{j}(indices) = beta_star;
         else
             beta{j} = zeros(p,1);
         end
         % Update the estimate of the covariance matrix         
         beta{j} = sparse(beta{j});
         W(:,j) = W*beta{j};
         W(j,:) = W(:,j)';
         W(j,j) = S(j,j);
    end
        
    var = W-W_old;
    var = var - diag(diag(var));
    var = triu(var);
    meanvariation = sum(sum(abs(var)))/pp;
    W_old2 = W_old - diag(diag(W_old));  
    W_old2 = triu(W_old2);
    mean_old = sum(sum(abs(W_old2)))/pp;
    % Test if tolerance has been reached
    if meanvariation <= tol * mean_old 
        if nargout > 1
            % Compute Omega
            for j=1:p
                O22 = 1/(S(j,j) - W(j,:)*beta{j});
                Omega(:,j) = -beta{j}*O22;
                Omega(j,:) = Omega(:,j)';
                Omega(j,j) = O22;
            end
        end
        return;
    end  
end

warning('Warning: Maximum number of iterations reached');


end
