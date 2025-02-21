function Tcyl = getTAGVcyl( pan, tilt, config )
% pan, tilt : in rad
% config : [l1, l2, h]
% l1 : a length from cylinder origin to tilt joint
% l2 : a length from AGV origin to tilt joint at tilt 0
% h : a height from tilt joint to cylinder origin

if nargin < 3
    config = [321.1, 143.1, 110]; % 321.1
end

l1 = config(1);
l2 = config(2);
h = config(3);

cp = cos(pan);
sp = sin(pan);
ct = cos(-tilt);
st = sin(-tilt);

% fix
T_A_P = [cp, -sp, 0, 0 ; sp, cp, 0, 0 ; 0, 0, 1, 0 ; 0, 0, 0, 1];
% optim
T_P_T0 = [eye(3), [-143.1, 0, 0]'; 0, 0, 0, 1];
% fix
L_P_T0 = norm( T_P_T0(1:3, 4) );
% motor movement
mtrMove = -tan(tilt) * L_P_T0;
T_T0_T1 = eye(4);
T_T0_T1(3, 4) = mtrMove;

% fix
T_T1_T2 = [ct, 0, st, 0 ; 0, 1, 0, 0 ; -st, 0, ct, 0 ; 0, 0, 0, 1];

% optim
T_T2_CYL = [0, -1, 0, 321.1 ; -1, 0, 0, 0 ; 0, 0, -1, 110 ; 0, 0, 0, 1 ];

Tcyl = T_A_P * T_P_T0 * T_T0_T1 * T_T1_T2 * T_T2_CYL;



