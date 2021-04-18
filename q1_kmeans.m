rng default; % For reproducibility
%generate 4 clusters of data 
X = [randn(100,3)*0.75+ones(100,3)*3;
    randn(100,3)*0.5-ones(100,3)*2
    randn(100,3)*0.75-ones(100,3)*4
    randn(100,3)*0.75];
%plot data 
figure;
plot(X(:,1),X(:,2),'.');
title 'Randomly Generated Data';
%call own implementation of k means 
k = 4;
[cluster_idx,centers] = mykmeans_implem(X,k);

%plot clusters with centroids
figure;
plot(X(cluster_idx==1,1),X(cluster_idx==1,2),'r.','MarkerSize',12)
hold on
plot(X(cluster_idx==2,1),X(cluster_idx==2,2),'g.','MarkerSize',12)
plot(X(cluster_idx==3,1),X(cluster_idx==3,2),'b.','MarkerSize',12)
plot(X(cluster_idx==4,1),X(cluster_idx==4,2),'y.','MarkerSize',12)
plot(centers(:,1),centers(:,2),'kx',...
     'MarkerSize',15,'LineWidth',3) 
legend('Cluster 1','Cluster 2','Cluster 3','Cluster 4','Centroids','Location','NW')
%legend('Cluster 1','Cluster 2','Cluster 3','Centroids','Location','NW')
title 'Cluster Assignments and Centroids'
hold off



%spectral k-means implementation
Y = X*X'; %compute Y
[V,D] = eig(Y);%perform eigen value decomposition
[d,ind] = sort(diag(D),'descend');
spectral_X = V(:,ind(1:k,:));
%spectral_X = Vs';
[cluster_idx,centers] = mykmeans_implem(spectral_X,k);


%plot clusters with centroids
figure;
plot(X(cluster_idx==1,1),X(cluster_idx==1,2),'r.','MarkerSize',12)
hold on
plot(X(cluster_idx==2,1),X(cluster_idx==2,2),'g.','MarkerSize',12)
plot(X(cluster_idx==3,1),X(cluster_idx==3,2),'b.','MarkerSize',12)
plot(X(cluster_idx==4,1),X(cluster_idx==4,2),'y.','MarkerSize',12)
legend('Cluster 1','Cluster 2','Cluster 3','Cluster 4','Location','NW')
%legend('Cluster 1','Cluster 2','Cluster 3','Centroids','Location','NW')
title 'Cluster Assignments and Centroids'
hold off


%function for k means implementation
function [data_cluster,centers] = mykmeans_implem(X,k)
%find data size
data_size = size(X,1);
%randomly select any data point as center 
init_centers = randperm(data_size,k);
old_centers = zeros(k,size(X,2));
for i = 1:k
    old_centers(i) = X(init_centers(i));
end

%create vector cluster of data points
data_cluster = zeros(size(X,1),1);
centers(:,:) = old_centers;%set centroid to old centroid 
count = 0;
while ~isequal(old_centers,centers) || count == 0
count = 1;
old_centers(:,:) = centers(:,:);
for i = 1:data_size
    dist = zeros(k,1); %initialize distance vector from each centroid 
    for j = 1:k
        dist(j) = norm(old_centers(j)-X(i))^2;
    end    
    [M,I] = min(dist); %Get minimum distance centroid 
    data_cluster(i,1) = I;

end
%calculate new centers
for j = 1:k
    idx = find(data_cluster == j);
    centers(j,:) = mean(X((idx),:));    
end
end
end