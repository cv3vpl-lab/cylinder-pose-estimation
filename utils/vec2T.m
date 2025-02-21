function T = vec2T(v)

R = rotvec2mat3d( v(1:3) );
t = v(4:6);
T = [R, t'; 0 0 0 1];