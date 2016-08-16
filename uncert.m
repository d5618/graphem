function [v_mean, v_space, S] = uncert(avail,miss,iptrn,Cf,D,w)
% uncert.m
% Computes the variance of the space average of the prediction from GraphEM
% Input: avail, miss, iptrn =  pattens of missing values (output of GraphEM)
%        Cf = residual covariance matrix of the model (output of GraphEM)
%              note: must use the unitless C (Cf)!!!
%        D = scale factors for matrix C (output of GraphEM)
%        w = vector of weights used to compute a weighted space average
%        of the temperature field (generally cos(latitude))
% Output: v_mean = variance of the temperature average over time predicted by the
% multivariate normal model.
%         v_space = spatial variance at each time
%         S = covariance

npatterns = length(miss);
nyears = length(iptrn);
ntemp = length(w);  % number of temperature locations
v1 = zeros(npatterns,1);
S1 = cell(npatterns,1);
v_space = zeros(nyears,ntemp);
v_mean  = zeros(nyears,1);

% Compute variance for each pattern of missing values
for i=1:npatterns
    [~,S] = ols(Cf,avail{i},miss{i});
    
    % Restore original scaling to residual covariance matrix
    pm    = length(miss{i}); % number of missing values in this pattern
    S     = S .* repmat(D(miss{i})', pm, 1) .* repmat(D(miss{i}), 1, pm);
    
    [ind_i,imiss] = intersect(miss{i},1:ntemp);
    wi = w(ind_i);
    
    % Restrict S to missing temperature points
    Si = S(imiss, imiss);
    % Si = S{i}(imiss,imiss);                           %
    S1{i} = diag(Si);
    v1(i) = wi.'*Si*wi;
end

% Store the variance for each year
for i=1:nyears
    pati = iptrn(i);
    v_mean(i) = v1(pati);
    [~,~,itemp] = intersect(miss{pati},1:ntemp);
    v_space(i,itemp) = S1{pati};
end


end
