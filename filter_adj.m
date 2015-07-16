% Filter the adjacency matrix "adj" so that only 
% points at distance <= dmax remain

function Af = filter_adj(adj,lonlat, dmax)

    adj_d = dMat(lonlat,dmax);
    Af = adj .* adj_d;

end