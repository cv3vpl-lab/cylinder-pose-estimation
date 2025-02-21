function figresize(sz)

if nargin < 1
    sz = [1000, 700];
end

f1 = gcf;
p1 = f1.Position;
p1(3:4) = sz;
f1.Position = p1;