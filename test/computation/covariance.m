% OBJECTIVE OF FILE HERE
% Data derived from Linear Algebra with Applications, 9th ed., Steven J. Leon

clear

% Raw data matrix R; each col represents a grade category/variable (assignments,
% exams, final) and each row a sample student.
R = [198 200 196; 
    160 165 165;
    158 158 133;
    150 165 91;
    175 182 151;
    134 135 101;
    152 136 80];

% Take average along col vectors of R
a = mean(R, 1);

% Adjust scores so mean is zero
X = R - a;

% Compute cosine of first two col vectors of X
%x1 = X(:, 1);
%x2 = X(:, 2);
%(x1.' * x2)/(norm(x1) * norm(x2));

% Compute covariance matrix with and without using cov()
format shortG % avoid using power notation
n = size(X, 1);
S = ((X.') * X)/(n-1);
%disp(S);
s = cov(X);
%disp(s);

% Both S and s will be the same