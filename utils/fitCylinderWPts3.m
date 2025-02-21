function [cylParams, fvals] = fitCylinderWPts3( Pts3, cylRadius )
% Pts3 : 3 * n matrix of n point coordinates
% cylParams : [cylorg, cyldir]

assert( size(Pts3, 1) == 3 );

ctr = mean(Pts3, 2);
coeff = pca(Pts3');

% initial radial direction
rdir = coeff(:, 3);

% assumption : the cylinder axis is further from the camera than the cylinder surface
% points 

if rdir(3) < 0
    rdir = -rdir;
end
assert( rdir(3) > 0 );

linePts = [ctr, ctr + rdir];
d = getDistPts3ToLine( Pts3, linePts );
[~, i] = min(d);

d2surface = norm(ctr - Pts3(:, i));

[K, D] = estCurvatures( Pts3 );
cylorg = ctr + rdir * (cylRadius - d2surface);
cyldir = K(:, 1, i);

cylParams0 = [cylorg', cyldir'];

opt = optimset('Display', 'none', 'TolFun', 1e-5, 'TolX', 1e-5, 'MaxFunEvals', 1e5, 'MaxIter', 1e5);

f = @(cylParams)dist(cylParams, Pts3, cylRadius);
fval0 = dist(cylParams0, Pts3, cylRadius);

[cylParams, fval] = fminsearch( f, cylParams0, opt );
cylParams = [cylParams0; cylParams];
fvals = [fval0, fval];
% [fval0, fval]
end

function v = dist( cylParams, Pts3, cylRadius )
    ln = [cylParams(1:3)', cylParams(1:3)' + cylParams(4:6)'];
    d = getDistPts3ToLine( Pts3, ln );
    v = d - cylRadius;
    v = v * v';
end

