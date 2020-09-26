% Eigenfaces--compute eigenfaces using PCA
%  Facedata.mat contains 56x46 grayscale images of 40 people/classes, where
%   each person has 10 images.
%  Variable facedata is 40 (people) x 10 (images).
% Note: subplot(nrows, ncols, plot_number)


% Setup: Load face data from file and typecast matrices to double
clear
load('Facedata.mat');
colormap(gray);
for i=1:40
    for j=1:10
        facedata{i, j} = double(facedata{i, j});
    end
end



% a. 1) Plot mean image of person 1
mean1 = zeros(56, 46);

% Add up face pixel values at corresponding positions
for j=1:10
    mean1 = mean1 + facedata{1, j};
end
mean1 = (1/10) * mean1;
subplot(1, 2, 1);
imagesc(mean1);
title('Mean Face');



% a. 2) Plot eigenfaces of person 1
% Matrix X will be 10x2576, where the rows are the mean-centered images 
% (reshaped to 1x2576) and the columns the attributes/variables/individual 
% pixel values.
X = [reshape((facedata{1, 1} - mean1)', 1, []);
     reshape((facedata{1, 2} - mean1)', 1, []);
     reshape((facedata{1, 3} - mean1)', 1, []);
     reshape((facedata{1, 4} - mean1)', 1, []);
     reshape((facedata{1, 5} - mean1)', 1, []);
     reshape((facedata{1, 6} - mean1)', 1, []);
     reshape((facedata{1, 7} - mean1)', 1, []);
     reshape((facedata{1, 8} - mean1)', 1, []);
     reshape((facedata{1, 9} - mean1)', 1, []);
     reshape((facedata{1, 10} - mean1)', 1, []);
];

S = cov(X);% compute covariance matrix (2576*2576)

% Get eigenfaces
[U, V] = eig(S);% NOTE: max eig vals are at the bottom right

% Plot eigenfaces: Since I only have n=10 samples, and the dimensionality
% d=2576=56*46 (obviously n<<d), and, assuming linearly indpt data samples,
% we have at MOST n-1 eigenvectors.

% eig vectors to plot: 2576, ..., 2568
% TODO: loop over 2567-2568 and 
% TODO: expand subplot
max_eig = U(:, 2568);% get correspoding col vector
max_eig = reshape(max_eig, [46,56])';% reshape and transpose to get og shape
subplot(1, 2, 2);
imagesc(max_eig);
title('V1');
