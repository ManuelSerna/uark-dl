% Individual components of PCA

% Rows are individual experiments/samples.
% Columns represent the variables.
data = [2 1;
        2 2;
        2 3;
        4 3;
        5 3;
        6 4];

% Take mean
m = mean(data, 1);

% Mean center data
X = data-m;

% Get covariance matrix
S = cov(X);

% Compute eigs
[V, D] = eig(S);
