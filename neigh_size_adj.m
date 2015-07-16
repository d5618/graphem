% Computes an adjacency matrix where the k closest neighbors are adjacents

function adj = neighMat(lonlat,k)

    p = size(lonlat,1);  % Number of points
    d = zeros(p,p);
    adj = eye(p,p);
    
       
    [X,Y] = meshgrid(lonlat(:,1)*pi/180,lonlat(:,2)*pi/180);
    d = greatCircleDistance(Y,X',Y',X);
    
    for i=1:p
        [s,ind] = sort(d(i,:));
        adj(i,ind(1:k)) = 1;
        adj(ind(1:k),i) = 1;
    end
end