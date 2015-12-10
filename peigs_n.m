% Compute the positive eigenvalues of a matrix A

function [V,d,r] = peigs_n(A)

    [V1,d1] = eig(A);
    d1 = diag(d1);
    % Find real eigenvalues
    d1_real_idx = find(abs(imag(d1)) < eps);
    d1_real = d1(d1_real_idx);
    V1_real = V1(:,d1_real_idx);
    % Find positive eigenvalues
    d_idx = find(d1_real > eps);
    d = d1_real(d_idx);
    V = V1_real(:,d_idx);
    
    % Gets rid of small imaginery parts (if any)
    d = real(d);
    V = real(V);
    
    % ensure that eigenvalues are monotonically decreasing
    [d, I] = sort(d, 'descend');
    V = V(:, I);
    
    r = length(d_idx);

end