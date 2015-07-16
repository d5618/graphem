% Computes an adjacency matrix where two points are 
% adjacent if and only if they are at distance <= d_max

function adj = distance_neigh_adj(lonlat,d_max)

    p = size(lonlat,1);  % Number of points
    d = zeros(p,p);
    adj = eye(p,p);
    
       
    [X,Y] = meshgrid(lonlat(:,1)*pi/180,lonlat(:,2)*pi/180);
    d = greatCircleDistance(Y,X',Y',X);
    
    adj = d <= d_max;

end