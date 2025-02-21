function cylParams = applyCylParamsPrior(cylParams, cylPts3)
%
% In our scenario, the cylinder upward (y-axis) direction is similar to the
% y axis of the camera
% Make the cylinder origin close to the laser grid point of the lowest y-value 
    origin = cylParams(1:3)';     % [o_x; o_y; o_z]
    direction = cylParams(4:6)';    % [d_x; d_y; d_z]

    % 카메라 y축과 일치하도록, 방향의 y 성분이 음수이면 반전합니다.
    if direction(2) < 0
        direction = -direction;
    end

    % cylPts3에서 y값이 최소인 점을 찾습니다.
    [min_y, ~] = min(cylPts3(2, :));
    % laser grid의 최소 y값
    y_min = min_y;

    % 원통 기준점의 y 값와 방향의 y 성분을 사용해 이동 스칼라 t를 구합니다.
    if abs(direction(2)) < eps
        t = 0;  % d_y가 0에 가까우면 보정하지 않음
    else
        t = (y_min - origin(2)) / direction(2);
    end

    % 새로운 원통 기준점 계산: 원래 기준점에 t*direction 만큼 이동
    new_origin = origin + t * direction;

    % 결과 재조합: 방향은 그대로, 기준점은 new_origin로 대체
    cylParams = [new_origin', direction'];
end