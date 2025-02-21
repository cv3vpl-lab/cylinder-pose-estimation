function [T, fval] = fitCylinderWPts3sAngs( Pts3s, angs, cylRadius )
% angs{i}(pan, tilt)는 실린더 좌표계이므로 getTAGVcyl를 사용해 AGV 좌표계로 변환
% 변환된 AGV 좌표계로부터 얻은 실린더 모델과          (=AGV 좌표계)
% Pts3s로부터 피팅한 실린더 모델을 1대1 매칭(최적화)     (=카메라 좌표계)
% 최적화가 끝나면(Pts3s와 angs{i}(pan, tilt)가 매칭, 즉 카메라와 AGV좌표계간 매칭이 된 것이고)
% 이로부터 T_C1_AGV(AGV에서 카메라로 변환가능한 관계를 알 수 있음)
% 이후 AGV의 팬틸트를 T_C1_AGV를 적용하여 계산 후 정확한 실린더의 중심축을 알 수 있음 
% T_C1_cyl = T_C1_AGV * getTAGVcyl( angles(i, 1), angles(i, 2));
% getTAGVcyl(실린더 => AGV로), T_C1_AGV(AGV => 카메라로(이미지)), 즉 T_C1_cyl(실린더 => 카메라) 
% 따라서, 실린더 좌표계의 angs(팬틸트)를 카메라 좌표계(이미지)상에 축을 그리기 위함 

% Pts3s : cell array of Pts3; 
% Pts3 : 3 * n matrix of n point coordinates
% angs : angles in radian
% T : T_cam1_AGV

nAngles = numel(angs);
assert( nAngles >= 2 );
assert( numel(Pts3s) == nAngles );
assert( all( cellfun( @(x) size(x, 1)==3, Pts3s )) );

cylParams = cell(nAngles, 1);
fval = cell(nAngles, 1);
TAGVcyls = cell(nAngles, 1);
T1cyls = cell(nAngles, 1);
cylorgs_AGV = zeros(3, nAngles);
lns = cell(nAngles, 1);

for i = 1:nAngles
    % AGV-based cylinder pose
    TAGVcyls{i} = getTAGVcyl( angs{i}(1), angs{i}(2) );
    cylorgs_AGV(:, i) = TAGVcyls{i}(1:3, 4);

    % initially estimated cam-based cylinder pose
    [cylParams{i}, fval{i}] = fitCylinderWPts3( Pts3s{i}, cylRadius );   
    cylParams{i} = applyCylParamsPrior(cylParams{i}, Pts3s{i});
    lns{i} = [cylParams{i}(1:3)', cylParams{i}(1:3)' + cylParams{i}(4:6)'];
end

T1cyls{1} = TAGVcyls{1};
for i = 2:nAngles
    T1cyls{2} = TAGVcyls{1} \ TAGVcyls{i};
end

%%%%%%%%%%%%%%%%%%%%%%%%%%
%     initial AGV pose : T0
%
% the cylinder origin at angle 1 should be the same (cylParams{1}, TAGVcyls{1})
% cylinder axis of cylParams{1} is aligned with the y-axis of TAGVcyls{1}
% the vector direction of the cylinder origins 1 & 2 should be similar.
p1 = TAGVcyls{1}(1:3, 4);
p2 = TAGVcyls{2}(1:3, 4);
ep1 = cylParams{1}(1:3)';
ep2 = cylParams{2}(1:3)';
% the vector of the cylinder origins w.r.t. AGV
d12 = p2 - p1; 
nd123 = cross(TAGVcyls{1}(1:3, 2), d12);
nd123 = nd123 / norm(nd123);
% the vector of the cylinder origins initially estimated 
ed12 = ep2 - ep1;  
end123 = cross(cylParams{1}(4:6)', ed12);
end123 = end123 / norm(end123);

R_C1_AGV = [ cylParams{1}(4:6)', end123, cross(cylParams{1}(4:6)', end123) ] ...
         / [ TAGVcyls{1}(1:3, 2), nd123, cross(TAGVcyls{1}(1:3, 2), nd123) ];
t = ep1 - R_C1_AGV * p1;

T0 = [R_C1_AGV, t; 0, 0, 0, 1];
agvPose0 = T2vec(T0);

f = @(agvPose)dist( agvPose, TAGVcyls, Pts3s, cylRadius);
fval0 = dist(agvPose0, TAGVcyls, Pts3s, cylRadius);

% optimization
opt = optimset('Display', 'none', 'TolFun', 1e-5, 'TolX', 1e-5, 'MaxFunEvals', 1e5, 'MaxIter', 1e5); % 'PlotFcns', @optimplotfval
[agvPose, fval] = fminsearch( f, agvPose0, opt );

T = vec2T(agvPose);
[fval0, fval]
end

function v = dist( agvPose, TAGVcyls, Pts3s, cylRadius )
    T = vec2T(agvPose);
    nAngles = length(TAGVcyls);
    v = 0;
    for i = 1:nAngles
        T_C1_cyl = T * TAGVcyls{i};
        dy = T_C1_cyl(1:3, 2);
        ln = [T_C1_cyl(1:3, 4), T_C1_cyl(1:3, 4) + dy];
        d = getDistPts3ToLine( Pts3s{i}, ln );
        vi = d - cylRadius;
        v = v + (vi * vi') / length(vi);
    end
end



