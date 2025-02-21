function [K, L] = estCurvatures(Pts3)

n = size(Pts3, 2);

idx = knnsearch( Pts3', Pts3', 'K', 20 );

K = zeros(3, 2, n);
L = zeros(2, n);
for i = 1:n
    pln = fitplane( Pts3(:, idx(i, :)) );
    lc = createLocCoordSys( pln(1:3) );
    coeffs = fitquadsurf( Pts3(:, idx(i, :)), lc );
    [V, D] = eig( [coeffs(1)*2, coeffs(2) ; coeffs(2), coeffs(3)*2] );
    K(:, :, i) = lc(:,1:2) * V;
    L(:, i) = diag(D);
end

end

function loccoords = createLocCoordSys( normal )
    z = normal;
    x = [1, 0, 0];
    if abs(z * x') > 0.9
        x = [0, 1, 0];
    end
    y = cross(z, x);
    x = cross(y, z);
    loccoords = [x', y', z'];
end

function coeffs = fitquadsurf( Pts3, loccoords )
    locPts3 = (Pts3' - mean(Pts3')) * loccoords;
    x = locPts3(:, 1);
    y = locPts3(:, 2);
    A = [x.^2, x.*y, y.^2, x, y];
    b = locPts3(:, 3);
    coeffs = A \ b;
end

