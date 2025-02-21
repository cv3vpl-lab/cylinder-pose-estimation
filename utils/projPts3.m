function pts2 = projPts3(pts3, K)

assert( size(pts3, 1) == 3);

pts2 = hom2cart(pts3' * K')';