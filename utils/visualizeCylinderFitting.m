function visualizeCylinderFitting(pts3, cylParams0, cylParams, cylRadius)
% pts3        : 3×N 행렬 (3D 점들)
% cylParams0  : 초기 원통 파라미터 [cylorg0, cyldir0] (1×6 벡터)
% cylParams   : 최종 원통 파라미터 [cylorg, cyldir] (1×6 벡터)
% cylRadius   : 원통 반경

    % 1) 3D 점 시각화
    % 매트랩 함수 내부에서 변수를 유지하기 위한 persistent 선언
    persistent figNum
    
    % 함수 최초 호출 시 figNum 초기화
    if isempty(figNum)
        figNum = 700;
    else
        figNum = figNum + 1;
    end
    
    % figure(figNum) 호출 -> 500, 501, 502 ... 순으로 열림
    figure(figNum);
    pcshow(pts3', 'MarkerSize', 20); 
    hold on;
    axis equal;
    title('3D Point Cloud + Cylinders');

    % 2) 초기 원통(연두색) 시각화
    drawCylinder3D(cylParams0, cylRadius, 150, [0,1,0], 0.3);

    % 3) 최종 원통(보라색) 시각화
    drawCylinder3D(cylParams, cylRadius, 150, [0.5,0,0.5], 0.3);

    % 범례 등 추가
    hLegend = legend('Point Cloud', 'Initial Cylinder', 'Final Cylinder');
    set(hLegend, 'TextColor', 'w');

end

function drawCylinder3D(cylParams, radius, height, colorVal, alphaVal)
% cylParams : [cylorg(1×3), cyldir(1×3)]
% radius    : 원통 반경
% height    : 원통 높이 (임의 설정)
% colorVal  : [R,G,B] 색상 (0~1)
% alphaVal  : 투명도 (0~1)

    % 분해
    cylorg = cylParams(1:3);    % 원통 기준점
    cyldir = cylParams(4:6);    % 원통 축 방향
    cyldir = cyldir(:) / norm(cyldir);  % 정규화(길이 1)

    % 원통을 parametric form으로 정의 (z축 방향을 축으로 가정)
    %  θ: [0, 2π], z: [0, height]
    nTheta = 50;
    nZ     = 50;
    [theta, zz] = meshgrid(linspace(0, 2*pi, nTheta), linspace(0, height, nZ));

    % 원통 중심축이 z축이라고 가정하면
    X = radius * cos(theta);
    Y = radius * sin(theta);
    Z = zz;

    % 이제 (X, Y, Z)를 원통 축(cyldir)에 맞춰 회전/이동
    % 1) 기본 축(z=[0,0,1]) -> cyldir 변환 위한 로컬 좌표계 생성
    zAxis = [0; 0; 1];
    if norm(cross(zAxis, cyldir)) < 1e-12
        % cyldir가 [0,0,1]과 거의 같은 방향이면 회전 불필요
        rotMat = eye(3);
    else
        % 로드리게스 공식 또는 vrrotvec2mat 등을 이용해 회전행렬 계산
        rotAxis = cross(zAxis, cyldir);
        rotAxis = rotAxis / norm(rotAxis);
        angle   = acos(dot(zAxis, cyldir));
        rotMat  = axang2rotm([rotAxis' angle]);  % R2020b 이후 가능
    end

    % 2) 회전 적용
    XYZ = [X(:), Y(:), Z(:)] * rotMat';
    % 3) 원통 기준점(cylorg)으로 평행이동
    XYZ = XYZ + cylorg;

    % surf로 그리기
    X_ = reshape(XYZ(:,1), size(X));
    Y_ = reshape(XYZ(:,2), size(X));
    Z_ = reshape(XYZ(:,3), size(X));

    surf(X_, Y_, Z_, 'FaceColor', colorVal, 'FaceAlpha', alphaVal, 'EdgeColor', 'none');
end

function R = axang2rotm(axang)
% axang = [axisX, axisY, axisZ, angle]
% MATLAB 내장 함수 vrrotvec2mat과 유사
    axis = axang(1:3);
    angle = axang(4);
    axis = axis / norm(axis);
    c = cos(angle);
    s = sin(angle);
    t = 1 - c;
    x = axis(1); y = axis(2); z = axis(3);
    R = [ t*x^2 + c   t*x*y - s*z  t*x*z + s*y
          t*x*y + s*z t*y^2 + c    t*y*z - s*x
          t*x*z - s*y t*y*z + s*x  t*z^2 + c   ];
end
