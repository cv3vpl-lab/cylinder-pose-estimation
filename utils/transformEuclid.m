function tpts = transformEuclid(T, pts)

% 3-D points
assert( size(pts, 1) == 3 );
assert( isequal(size(T), [4, 4]) );

tpts = hom2cart( cart2hom(pts') * T' )';