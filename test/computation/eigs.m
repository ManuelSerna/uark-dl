% Get eigenvalues and eigenvectors with examples

% For eig val = 4, eig vec has form [2 1]
% For eig val = -3, eig vec has form [-1 3]
%A = [ 3 2; 3 -2 ];

% For eig val = 5, eig vec has form [1 2]
%A = [ 1 2; 2 4];

% For eig val = 3, eig vec has form [2 1]
%A = [ 4 -2; 1 1 ];

% For eig val = 2, eig vec has form [1 1]
% For eig val = 3, eig vec has form [4 3]
A = [ 6 -4; 3 -1];

% Use eig() function
[V, D] = eig(A);
%disp(D);% matrix where diagonals are eigenvalues
%disp(V);% matrix with corresponding eigenvectors
