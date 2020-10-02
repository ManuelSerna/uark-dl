% Individual components of PCA

% Rows are individual experiments/samples.
% Columns represent the variables.
%{
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
%}


X = [2 2 2 4 5 6;
     1 2 3 3 3 4];
%mu = mean(X, 2);
%X = X - mu;
%S = (1/5)*X*X';
%[V, D] = eig(S);

Xhat = [1.992 2.397 2.801 4.388 5.182 6.380;
        1.879 2.104 2.310 3.120 3.525 4.136];

mse = 0;
for i=1:6
    A = X(:,i)-Xhat(:,i);
    mse = mse + A'*A;
end

mse = mse * (1/6);
