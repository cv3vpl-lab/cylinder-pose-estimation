function v = T2vec(T)

R = T(1:3, 1:3);
v = [rotmat2vec3d(R), T(1:3, 4)'];