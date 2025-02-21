function [X, Y, Z] = getCylinder(ln, pts3, cylRadius, lengthCyl)
    % getCylinder:
    %   ln        : 3×2 행렬, 실린더 축의 양끝 점 (각 열이 한 점)
    %   pts3      : 3×N 행렬, 실린더가 덮어야 하는 포인트 집합
    %   cylRadius : 실린더의 반지름 (예: 45)
    %   lengthCyl : 실린더의 전체 길이 (예: 200; y축 기준)
    %
    % 반환:
    %   X, Y, Z : 실린더 표면의 좌표 행렬
    %
    % [동작 설명]
    % 1. pts3의 중심을 구하고, ln의 축(직선)에 직교투영하여
    %    평행 이동 기준점을 구합니다.
    % 2. 기본 실린더는 y축 방향으로 생성되며, y값이 -lengthCyl/2 ~ lengthCyl/2로
    %    설정되어 실린더의 중간이 원점이 되도록 합니다.
    % 3. y축 [0;1;0]이 ln의 방향과 일치하도록 회전한 후, 평행 이동합니다.

    % ln의 두 점 추출
    p1 = ln(:,1);
    p2 = ln(:,2);
    
    % ln의 방향(단위 벡터) 계산
    d = p2 - p1;
    d = d / norm(d);
    
    % pts3의 중심 계산
    center = mean(pts3, 2);
    
    % pts3의 중심을 ln의 축 상에 직교 투영 (평행 이동은 ln의 축 방향으로만)
    mid_proj = p1 + (d' * (center - p1)) * d;
    
    % mid_proj를 중심으로, d 방향을 축으로 하는 실린더 좌표 생성
    [X, Y, Z] = createCylinderFromAxis(mid_proj, d, cylRadius, lengthCyl, 50);
end

function [X, Y, Z] = createCylinderFromAxis(mid, d, cylRadius, lengthCyl, nFacets)
    % 기본 cylinder 좌표 생성 (MATLAB 내장 cylinder 함수 사용)
    [x, y, z] = cylinder([1 1], nFacets);
    
    % 기본 cylinder는 z축이 길이 방향으로 생성되므로, y축 방향으로 사용하기 위해 좌표 교환
    temp = y;
    y = z;      % y축을 길이 방향으로
    z = temp;   % z축은 원래 y값으로
    
    % y값을 -lengthCyl/2 ~ lengthCyl/2로 확장 (즉, 중심이 0)
    y = (y - 0.5) * lengthCyl;
    
    % 반지름 적용
    x = x * cylRadius;
    z = z * cylRadius;
    
    % 3×N 점들로 정리 (각 열이 한 3D 점)
    pts = [x(:)'; y(:)'; z(:)'];
    
    % y축 [0;1;0]을 d 방향으로 회전시키기 위한 회전행렬 계산
    R = rotationToAlignWithY(d);
    
    % 회전 적용
    pts_rot = R * pts;
    
    % 회전 후, 생성된 실린더는 y축 방향의 중심(0)이 원점에 있으므로,
    % mid (즉, ln의 축 상에 투영된 pts3의 중심)로 평행 이동
    pts_trans = pts_rot + mid;
    
    % 원래의 meshgrid 형태로 재배열하여 결과 반환
    X = reshape(pts_trans(1,:), size(x));
    Y = reshape(pts_trans(2,:), size(y));
    Z = reshape(pts_trans(3,:), size(z));
end

function R = rotationToAlignWithY(v)
    % rotationToAlignWithY:
    %   y축 [0;1;0]을 주어진 벡터 v(단위벡터)와 일치하도록 회전하는
    %   회전행렬을 반환합니다.
    v = v / norm(v);
    y_axis = [0; 1; 0];
    
    % 이미 정렬되어 있다면 단위 행렬 반환
    if norm(v - y_axis) < 1e-6
        R = eye(3);
        return;
    end
    
    % 정반대인 경우 180도 회전
    if norm(v + y_axis) < 1e-6
        R = rodrigues([1; 0; 0], pi);
        return;
    end
    
    % 회전축과 회전각 계산
    axis = cross(y_axis, v);
    axis = axis / norm(axis);
    angle = acos(dot(y_axis, v));
    
    % Rodrigues 공식으로 회전행렬 계산
    R = rodrigues(axis, angle);
end

function R = rodrigues(axis, theta)
    % rodrigues:
    %   axis : 회전축 (단위 벡터, 3×1)
    %   theta: 회전각 (라디안)
    %   R    : 3×3 회전행렬 (Rodrigues 공식 사용)
    ux = axis(1);
    uy = axis(2);
    uz = axis(3);
    c = cos(theta);
    s = sin(theta);
    
    R = [ c + ux^2*(1-c),      ux*uy*(1-c)-uz*s,  ux*uz*(1-c)+uy*s;
          uy*ux*(1-c)+uz*s,    c + uy^2*(1-c),    uy*uz*(1-c)-ux*s;
          uz*ux*(1-c)-uy*s,    uz*uy*(1-c)+ux*s,  c + uz^2*(1-c)];
end
