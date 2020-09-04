% Ovarian cancer dataset analysis using PCA
% Code credit: Steve Brunton

clear all, close all, clc

load ovariancancer;% load data

% Matrix "obs" corresponds to 216 patients who all had 4000
% genetic markers measured. Our goal is to find which genetic markers most
% correlate to whether or not the patients do/don't have ovarian cancer.
% Matrix grp has labels.

% We can take the SVD of our data. In this case, U is 216 x 216, and V is
% 4000 x 4000. V has the eigenvectors corresponding to the genetic markers.
[U, S, V] = svd(obs, 'econ');

% Plot the three dominant genetic markers
figure
subplot(1, 2, 1)
semilogy(diag(S), 'k-o', 'LineWidth', 2.5)
set(gca, 'FontSize', 15), axis tight, grid on

% Good idea to plot singular values (eigenvalues) to see what eigenvectors
% are the strongest/have the max variability.
subplot(1, 2, 2)
plot(cumsum(diag(S))./sum(diag(S)), 'k-o', 'LineWidth', 2.5)
set(gca, 'FontSize', 15), axis tight, grid on
set(gcf, 'Position', [1400 100 3*600 3*250])

figure, hold on
for i=1:size(obs, 1)
    x = V(:,1)'*obs(i,:)';
    y = V(:,2)'*obs(i,:)';
    z = V(:,3)'*obs(i,:)';
    if(grp{i}=='Cancer')
        plot3(x, y, z, 'rx', 'LineWidth', 3);
    else
        plot3(x, y, z, 'bo', 'LineWidth', 3);
    end
end

xlabel('PC1'), ylabel('PC2'), zlabel('PC3')
view(85, 25), grid on, set(gca, 'FontSize', 15)
set(gcf, 'Position', [1400 100 1200 900])
