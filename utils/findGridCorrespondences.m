function [cgp1, cgp2, cgpi] = findGridCorrespondences(gp1, gp2)
% gp1 : grid points 1 ; n * 4 matrix ( x, y, x index, y index )
% gp2 : grid points 2
% cgp1/2 : corresponding points 1/2
% cgpi : corresponding point index

gpi1 = gp1(:, 3:4);
gpi2 = gp2(:, 3:4);
n = size(gpi1, 1);

cgp1 = [];
cgp2 = [];
cgpi = [];
for i = 1:n
    idx = find( all( gpi2 == gpi1(i, :), 2 ) );
    if isempty(idx), continue; end

    cgp1 = [cgp1; gp1( i, 1:2 )];
    cgp2 = [cgp2; gp2( idx, 1:2 )];
    cgpi = [cgpi; gpi1(i, :)];
end