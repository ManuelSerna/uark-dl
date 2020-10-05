%% Eigenfaces: compute eigenfaces using PCA
% File 'Facedata.mat' contains 56x46 grayscale images of 40 people/classes,
%  where each person has 10 images.
% Variable facedata is 40 (people) x 10 (images).
% We will only work with the faces of person 1.
% Author: Manuel Serna-Aguilera


%% Setup: Load face data from file
clear
tic % start timer
load('Facedata.mat');

% Use im2double to convert all face images to double (in range [0,1])
for i=1:40
    for j=1:10
        facedata{i,j} = im2double(facedata{i,j});% grows due to precision, but is acceptable in this case
    end
end
clear i

% Store raw face data of person 1 as col vectors
data = zeros(2576,10);
for j=1:10
    data(:,j) = facedata{1,j}(:);
end

% Compute mean image of person 1
mean = zeros(2576,1);
for j=1:10
    mean = mean + data(:,j);
end
mean = (1/10) * mean;

% Build data matrix X, where X(:,j) has the mean centered data for face j
X = zeros(2576, 10);
for j=1:10
    X(:, j) = data(:,j)-mean;
end

clear data
clear j


%% Part A: Plot eigenfaces of covariance matrix
figure(1) % first figure for first set of eig faces
colormap(gray);

% Plot mean image of person 1
subplot(2, 5, 1);
imagesc(reshape(mean, [56,46]));
title('Mean Face');

% Compute 2576*2576 covariance matrix S and get its eigs
S = cov(X');
[eigvectors, eigvals] = eig(S);% NOTE: max eig vals are at bottom of diagonal
%Z = norm(eigvectors(:,10));% double check eigvectors of cov matrix are normal

% Plot eigenfaces: Since I only have n=10 samples, and the dimensionality
%  d=2576=56*46 (obviously n<<d), and, assuming linearly indpt data samples,
%  we have at MOST n-1 eigenvectors.
p = 1;
j = 2576;% start with max eig face
while j > 2567
    v = reshape(eigvectors(:, j), [56,46]);
    subplot(2, 5, p+1);
    imagesc(v);
    title(strcat('V',int2str(p)));
    
    % Move on to next eig face
    p = p + 1;
    j = j - 1;
end

clear j
clear p
clear v
clear S
clear eigvectors
clear eigvals


%% Part B
% Time for plotting first set of eig faces
time_pt_b = toc;
fprintf('First timing (B): %f\n', time_pt_b); % typically clocks at ~2.4 secs


%% Part C: Plot eigenfaces
figure(2)
colormap(gray);

G = cov(X);% compute inner product/Gram matrix G
[U, D] = eig(G);% get eigs of G
eigfaces = X*U;% get real eigenvectors

% Normalize eigenvectors to satisfy constraint ||v||=1
for i=1:10
    eigfaces(:,i) = eigfaces(:,i)/norm(eigfaces(:,i));
end
%Z = norm(eigfaces(:,10));% check that magnitude of eigface is 1

% Plot eig faces
subplot(2, 5, 1);% mean image first
imagesc(reshape(mean, [56,46]));
title('Mean Face');

p = 1;
j = 10;% start at max eigvector
while j > 1
    eigface = reshape(eigfaces(:, j), [56,46]);
    subplot(2, 5, p+1);
    imagesc(eigface);
    title(strcat('V',int2str(p)));
    
    % Move on to next eig face
    p = p + 1;
    j = j - 1;
end

clear eigface
clear j
clear p


%% E
fprintf('Second timing (E): %f\n', toc - time_pt_b);
clear time_pt_b


%% G: Project image 1 of person 1 onto eigenspace
figure(3)
colormap(gray);
subplot(2, 5, 1);% mean image first
imagesc(facedata{1,1});% plot target image
title('Original');

% Outer loop: estimation of image 1 with an increasing # of eigvectors
i = 10;% start at max eigvector
p = 1;% position in figure subplot
while i > 1
    % First, subtract mean image from image 1
    c = facedata{1,1}(:)-mean;
    
    est = mean;% initialize image 1 estimation matrix with mean face

    % Approximate face starting with max eigface 10
    j = 10;
    while j >= i
        weight = sum(c.*eigfaces(:,j));% add elements of element-wise product      
        est = est + (weight*eigfaces(:,j));% add eigvector*weight to estimation
        j = j - 1;% move on to (lesser) eigface
    end
    
    % Compute MSE
    mse = immse(facedata{1,1}(:), est);
    
    % Plot estimated face
    subplot(2, 5, p+1);
    imagesc(reshape(est, [56,46]));
    title(strcat('MSE=', num2str(mse)));
    
    % Move on to next eig face
    p = p + 1;
    i = i - 1;
end

clear c
clear est
clear i
clear j
clear mse
clear p
clear weight

