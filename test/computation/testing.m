% Test simple cases here
%A = [1 2;
%    3 4];
A = [1 2 3;
     4 5 6;
     7 8 9];

B = reshape(A', 1, []);% turn into row vector
%C = A(:);% turn into col vector
%a1 = A(:, 3); % grab third column vector of A

D = reshape(B, [3,3])';