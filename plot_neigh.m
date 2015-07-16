function plot_neigh(adj, lonlat, i)

    adj = adj - diag(diag(adj));
    neigh = find(adj(i,:));
    
    pts = lonlat(neigh,:);
    
    easy_scatter_map(pts);
    
    [X,Y] = m_ll2xy(lonlat(i,1), lonlat(i,2));
    scatter(X,Y,14,'red','filled');

end