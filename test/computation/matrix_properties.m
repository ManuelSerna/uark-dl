% Check matrix properties

%C = [1 -2 3; 2 -5 1; 1 -4 -7];
%rankC = rank(C);

A = [1 2 1; 2 4 2; 3 3 3];
B = [-1 2; 2 2];

%rankA = rank(A);
%rankB = rank(B);

%traceA = trace(A);
%traceB = trace(B);

%detA = det(A);
%detB = det(B);

[V, D] = eig(B);