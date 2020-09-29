%% Eigenfaces: compute eigenfaces using PCA
% File 'Facedata.mat' contains 56x46 grayscale images of 40 people/classes,
%  where each person has 10 images.
% Variable facedata is 40 (people) x 10 (images).
% Author: Manuel Serna-Aguilera


%% Setup: Load face data from file and typecast face matrices to double
tic
clear
load('Facedata.mat');
for i=1:40
    for j=1:10
        facedata{i, j} = double(facedata{i, j});
    end
end


%% Part A
figure(1) % first figure for first set of eig faces
colormap(gray);

% Compute mean image of person 1
mean1 = zeros(56, 46);
for j=1:10
    mean1 = mean1 + facedata{1, j};
end
mean1 = (1/10) * mean1;

% Plot mean image of person 1
subplot(2, 5, 1);
imagesc(mean1);
title('Mean Face');

% Build data matrix X, where col vector i is a reshaped image of person 1
X = zeros(2576, 10);
for i=1:10
    c = facedata{1,i}-mean1;% mean-center data
    X(:, i) = c(:);% reshape image into col vector
end

clear c
clear i
clear j

% Compute 2576*2576 covariance matrix S and get its eigs
S = cov(X');
[U, V] = eig(S);% NOTE: max eig vals are at bottom of diagonal

% Plot eigenfaces: Since I only have n=10 samples, and the dimensionality
%  d=2576=56*46 (obviously n<<d), and, assuming linearly indpt data samples,
%  we have at MOST n-1 eigenvectors.
p = 1;
counter = 2576;% start with max eig face
while counter > 2567
    v = reshape(U(:, counter), [56,46]);
    subplot(2, 5, p+1);
    imagesc(v);
    title(strcat('V',int2str(p)));
    
    % Move on to next eig face
    p = p + 1;
    counter = counter - 1;
end

clear p
clear v
clear counter

%% Part B
% Time for plotting first set of eig faces
time_pt_b = toc;
fprintf('First timing (B): %f\n', time_pt_b); % typically clocks at ~2.4 secs
clear time_pt_b


%% Part C
figure(2)
colormap(gray);

G = X'*X;% compute Gram matrix G
[U, V] = eig(G);% get eigs of G
U = normalize(U, 'range');% normalize eigvectors in U
eigfaces = X*U;% get eigenface vectors (each face is a col vector)

% Plot eig faces
subplot(2, 5, 1);% mean image first
imagesc(mean1);
title('Mean Face');

p = 1;
counter = 10;% start at max eigvector
while counter > 1
    v = reshape(eigfaces(:, counter), [56,46]);
    subplot(2, 5, p+1);
    imagesc(v);
    title(strcat('V',int2str(p)));
    
    % Move on to next eig face
    p = p + 1;
    counter = counter - 1;
end

clear p
clear v
clear counter


%% E
time_pt_e = toc;
fprintf('Second timing (E): %f\n', time_pt_e); % typically clocks at ~ secs
clear time_pt_e


%% G: TODO

