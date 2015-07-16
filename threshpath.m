%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% threshpath.m
%
% Computes a path of graphs by thresholding a sample correlation matrix
%
% Input:   - S = correlation matrix
%          - ind_T, ind_P = temperature and proxy indices
%          - levels = vector of sparsity levels (in %, ex: levels = [1,2,3])
%
% Output:  - adj = cell array of adjacency matrices with sparsity corresponding
%                  to levels in the TT and TP part of the graph. The PP part 
%                  is diagonal (no edges)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function adj = threshpath(S, ind_T, ind_P, levels)
		 
    pT = length(ind_T);  % Number of temperature points
    pP = length(ind_P);  % Number of proxies


    Stt = S(ind_T, ind_T);   % TT part of S
    Stp = S(ind_T, ind_P);   % TP part of S

    I = triu(ones(pT,pT)) - eye(pT); % Strict upper diagonal of pT x pT
    valsTT = Stt(logical(I));   % Upper diagonal entries of Stt
    valsTP = Stp(:);            % Entries of Stp

    sortedTT = sort(abs(valsTT),'descend');  % Sorted absolute values
    sortedTP = sort(abs(valsTP),'descend');

    % Compute thresholding level to obtain desired sparsity in TT and TP
    thresholdTT = sortedTT(ceil(levels(1)/100*length(sortedTT)));
    thresholdTP = sortedTP(ceil(levels(2)/100*length(sortedTP)));
    % Builds adjacency matrix
    A = zeros(pT+pP,pT+pP);
    A(ind_P,ind_P) = eye(pP);
    A(ind_T, ind_T) = abs(Stt) >= thresholdTT;
    A(ind_T, ind_P) = abs(Stp) >= thresholdTP;
    A(ind_P, ind_T) = A(ind_T, ind_P).';
    adj = logical(A);
end

% Test
% X = rand(10,5);
% S = corrcoef(X);
% ind_T = 1:2;
% ind_P = 3:5;
% levels = [10,30,50];
% adj = threshpath(S,ind_T, ind_P, levels);
