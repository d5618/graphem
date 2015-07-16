function [indices, screen_pp] = screen_proxies(Xcal, pi, pp, screen_radius, screen_corr, lonlat)

    dadj = dMat(lonlat, screen_radius);
    
    % Compute correlation matrix
    C = corr(Xcal);
    % Multiply by distance adjacency matrix
    C = C .* dadj;
    
    C_TP = C(1:pi, (pi+1):(pi+pp));
    
    maxC = max(C_TP);
    
    indices = find(maxC > screen_corr);
    
    indices = indices + pi;
    screen_pp = length(indices);
    indices = [1:pi, indices];
    

end