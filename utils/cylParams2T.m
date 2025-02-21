function T = cylParams2T(cylParams)

cylorg = cylParams(1:3)';
cyldir = cylParams(4:6)';

y = cyldir;
y = y / norm(y);
x = [1, 0, 0]';
z = cross(x, y); 
z = z / norm(z);
x = cross(y, z);
x = x / norm(x);

T = [x, y, z, cylorg; 0, 0, 0, 1];