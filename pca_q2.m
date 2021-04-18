%Part(1)
x1 = [0,0];
x2 = [-1,2];
x3 = [1,-2];
x4 = [3,-6];
x5 = [-3,6];
%Data for four points 
X_data = cat(1,x1,x2,x3,x4,x5);

%plot data 
figure;
hold on
plot(X_data(:,1),X_data(:,2),'.','MarkerSize',15,'LineWidth',3);
xlim([-8 8])
ylim([-8 8])
title 'Scatter plot of points';
hold off

%Part(2)
load('USPS.mat');
p = [10,50,100,200];
error = zeros(size(p,2),1);
count = 0 ;
for k = p  %for each value of number of principal component we need
    pc_components = mypca_implem(A,k); %call own implementation of pca using SVD
    %reconstruct from principal components
    reconstruct_A = A(:,:)*pc_components;
    reconstruct_A = reconstruct_A*pc_components';
    %reshape for image display
    A2 = reshape(reconstruct_A(1,:), 16, 16);
%save first data image 
    fname = strcat('first_', num2str(k) , '_component.jpeg');
    imwrite(A2',fname,'JPEG');
%save second data image    
    A2 = reshape(reconstruct_A(2,:), 16, 16);
    fname = strcat('second_', num2str(k) , '_component.jpeg');
    imwrite(A2',fname,'JPEG');
    count = count + 1 ;
%compute reconstruction error using Frobenius norm    
    error(count) = norm((A - reconstruct_A),'fro')^2;
end

plot(p,error)
xticks([10 50 100 200])
xlabel('Number of Principal Components');
ylabel('Reconstruction Error');

function [components] = mypca_implem(X,n_comp)
%center the data by subtracting mean
centered_data = X - mean(X);
%perform SVD on centerd data matrix 
[U,S,V] = svd(centered_data);
%columns of V are principal components
components = V(:,(1:n_comp));
end