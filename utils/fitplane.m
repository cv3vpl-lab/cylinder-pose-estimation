function P = fitplane(pts)
% P = fitplane(pts)
%
% pts : 3 * n matrix for n points
% P : 1 * 4 row vector representing a plane


% at least 3 points
assert( size(pts, 1) == 3);
assert( size(pts, 2) >= 3 );

cv = cov( pts' );
[V, ~] = eig(cv);
P = V(:, 1)';
P = [P, -mean(P * pts)];

