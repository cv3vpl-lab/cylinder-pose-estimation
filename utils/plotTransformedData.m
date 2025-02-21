function plotTransformedData(pts3, ln)
    % plotTransformedData 포인트 클라우드와 선을 변환하여 시각화하고,
    % 팬과 틸트 각도를 계산합니다.
    %
    % 입력:
    %   pts3       - 3xN 행렬. 포인트 클라우드의 좌표들.
    %   ln         - 3x2 행렬. 선을 정의하는 두 점의 좌표들.
    %   real_pan   - 실제 팬 각도 (도 단위). (선택 사항)
    %   real_tilt  - 실제 틸트 각도 (도 단위). (선택 사항)
    %
    % 출력:
    %   없음. 시각화된 플롯과 콘솔에 각도 출력.
    
    % 입력 인자 검증 및 기본값 설정
    if nargin < 3
        real_pan = NaN;
    end
    if nargin < 4
        real_tilt = NaN;
    end
    
    % 변환 행렬 (카메라 -> AGV)
    C2A = [
        0.0162   -0.9722   -0.2337   33.4075;
       -0.9999   -0.0163   -0.0014  121.1600;
       -0.0025    0.2337   -0.9723  404.7484;
             0         0         0    1.0000];
    
    %% 1. Figure 설정
    figure('Color', 'k'); % 배경을 검은색으로 설정
    hold on;
    xlabel('X', 'Color', 'w'); % X축 라벨, 흰색 글씨
    ylabel('Y', 'Color', 'w'); % Y축 라벨, 흰색 글씨
    zlabel('Z', 'Color', 'w'); % Z축 라벨, 흰색 글씨
    title('Original and Transformed Data', 'Color', 'w'); % 흰색 글씨 제목
    grid on; % 그리드 활성화
    view(3); % 3D 보기 설정
    
    % 축 표시
    ax = gca; % 현재 축 가져오기
    ax.XColor = 'w'; % X축 색상 흰색
    ax.YColor = 'w'; % Y축 색상 흰색
    ax.ZColor = 'w'; % Z축 색상 흰색
    axis on; % 축 보이기

    
    %% 2. 원래의 포인트 클라우드 시각화
    scatter3(pts3(1, :), pts3(2, :), pts3(3, :), 10, 'r', 'filled', 'DisplayName', 'Original Points');
    
    %% 3. 원래의 선 시각화 (길이 3배로 확장)
    % 원래 선의 방향 벡터
    original_direction = ln(:,2) - ln(:,1);
    original_length = norm(original_direction);
    original_direction_normalized = original_direction / original_length;

    original_extension = original_length * original_direction_normalized;
    
    % Extend the line equally on both sides
    new_start = ln(:,1) - 3*original_extension;
    new_end = ln(:,2) + 3*original_extension;

    % Set the extended line
    ln_extended_original = [new_start, new_end];
    
    scatter3(ln_extended_original(1,1), ln_extended_original(2,1), ln_extended_original(3,1), 100, 'y', 'filled', 'DisplayName', 'Original Point 1');
    scatter3(ln_extended_original(1,2), ln_extended_original(2,2), ln_extended_original(3,2), 100, 'y', 'filled', 'DisplayName', 'Original Point 2');
    plot3(ln_extended_original(1,:), ln_extended_original(2,:), ln_extended_original(3,:), 'y-', 'LineWidth', 2, 'DisplayName', 'Original Line');
    
    % Plot cylinder for original line
    [Xcyl, Ycyl, Zcyl] = createCylinder(ln_extended_original(:,1), ln_extended_original(:,2), 45, 200);
    set(gcf, 'Renderer', 'opengl'); % Set the figure renderer
    surf(Xcyl, Ycyl, Zcyl, ...
         'FaceColor', 'y', ...
         'FaceAlpha', 0.5, ...
         'EdgeColor', 'none', ...
         'DisplayName', 'Original Cylinder');
    %% 4. 포인트 클라우드 변환
    % pts3를 동차 좌표로 확장 (4xN)
    pts3_homog = [pts3; ones(1, size(pts3, 2))];
    
    % C2A 변환 행렬 적용 (4x4 * 4xN = 4xN)
    pts3_transformed_homog = C2A * pts3_homog;
    
    % 동차 좌표를 3D 좌표로 변환 (정규화)
    pts3_transformed = pts3_transformed_homog(1:3, :) ./ pts3_transformed_homog(4, :);
    
    %% 5. 변환된 포인트 클라우드 시각화
    scatter3(pts3_transformed(1, :), pts3_transformed(2, :), pts3_transformed(3, :), 10, 'g', 'filled', 'DisplayName', 'Transformed Points');
    
    %% 6. 선 변환 및 시각화 (길이 3배로 확장)
    % ln의 두 점을 동차 좌표로 확장 (4x2)
    ln_homog = [ln; ones(1, size(ln, 2))];
    
    % C2A 변환 행렬 적용 (4x4 * 4x2 = 4x2)
    ln_transformed_homog = C2A * ln_homog;
    
    % 동차 좌표를 3D 좌표로 변환 (정규화)
    ln_transformed = ln_transformed_homog(1:3, :) ./ ln_transformed_homog(4, :);
    
    % 변환된 선의 방향 벡터
    transformed_direction = ln_transformed(:,2) - ln_transformed(:,1);
    transformed_length = norm(transformed_direction);
    transformed_direction_normalized = transformed_direction / transformed_length;
    
    % Calculate the extension vector
    transformed_extension = transformed_length * transformed_direction_normalized;
    
    % Extend the line equally on both sides
    new_start = ln_transformed(:,1) - 3*transformed_extension;
    new_end = ln_transformed(:,2) + 3*transformed_extension;
    
    % Set the extended line
    ln_extended_transformed = [new_start, new_end];
    
    scatter3(ln_extended_transformed(1,1), ln_extended_transformed(2,1), ln_extended_transformed(3,1), 100, 'c', 'filled', 'DisplayName', 'Transformed Point 1');
    scatter3(ln_extended_transformed(1,2), ln_extended_transformed(2,2), ln_extended_transformed(3,2), 100, 'c', 'filled', 'DisplayName', 'Transformed Point 2');
    plot3(ln_extended_transformed(1,:), ln_extended_transformed(2,:), ln_extended_transformed(3,:), 'b-', 'LineWidth', 2, 'DisplayName', 'Transformed Line');

    % Plot cylinder for transformed line
    [XcylTrans, YcylTrans, ZcylTrans] = createCylinder(ln_extended_transformed(:,1), ln_extended_transformed(:,2), 45, 200);
    set(gcf, 'Renderer', 'opengl'); % Set the figure renderer
    surf(XcylTrans, YcylTrans, ZcylTrans, ...
     'FaceColor', 'b', ...
     'FaceAlpha', 0.5, ...
     'EdgeColor', 'none', ...
     'DisplayName', 'Original Cylinder');

    
    %% 7. 범례 설정
    legendHandle = legend('Location', 'best');
    set(legendHandle, 'Color', 'w'); % 범례 배경을 흰색으로 설정
   
end

function [X, Y, Z] = createCylinder(p1, p2, radius, lengthCyl)
    p1 = p1(:);
    p2 = p2(:);
    
    % Compute the direction vector and its unit vector
    v = p2 - p1;
    u = v / norm(v);
    
    % Generate unit cylinder
    [x, y, z] = cylinder([1 1], 50); % 50 facets around the circumference
    z = z * (lengthCyl / 2); % Scale z to the desired length
    
    % Scale x and y to the desired radius
    x = x * radius;
    y = y * radius;
    
    % Create points matrix
    points = [x(:), y(:), z(:)];
    
    % Compute rotation matrix to align with u
    R = rotationToAlignWithZ(u);
    
    % Apply rotation to the points
    points = points * R';
    
    % Translate the points to center at p1
    points = points + repmat(p1', size(points, 1), 1);
    
    % Reshape the points back to the original cylinder grid
    X = reshape(points(:,1), size(x));
    Y = reshape(points(:,2), size(y));
    Z = reshape(points(:,3), size(z));
end

function R = rotationToAlignWithZ(v)
    v = v / norm(v); % Normalize
    z_axis = [0; 0; 1];
    axis = cross(z_axis, v);
    if norm(axis) == 0
        % v is already aligned with z-axis
        R = eye(3);
        return;
    end
    axis = axis / norm(axis);
    angle = acos(dot(z_axis, v));
    R = rodrigues(axis, angle);
end

function R = rodrigues(axis, theta)
    % Rodrigues' rotation formula
    % axis: rotation axis (unit vector)
    % theta: rotation angle in radians
    % R: rotation matrix
    ux = axis(1);
    uy = axis(2);
    uz = axis(3);
    c = cos(theta);
    s = sin(theta);
    R = [c + ux^2*(1 - c)      ux*uy*(1 - c) - uz*s     ux*uz*(1 - c) + uy*s;
         uy*ux*(1 - c) + uz*s  c + uy^2*(1 - c)        uy*uz*(1 - c) - ux*s;
         uz*ux*(1 - c) - uy*s  uz*uy*(1 - c) + ux*s    c + uz^2*(1 - c)];
end