function pts = pointsStruct2mat(points)
    % points 구조체를 셀 배열로 변환
    pts0 = struct2cell(points);
    
    % 구조체가 3개의 필드를 가지는지 확인 (예: grid index, x, y)
    assert(size(pts0,1) == 3, '구조체는 3개의 필드를 가져야 합니다.');
    
    % 2번째와 3번째 필드는 좌표값으로 가정하고 숫자 배열로 변환
    ptsCoord = cell2mat(pts0(2:3, :));  % 2 x N 행렬
    
    % 첫 번째 필드는 grid index로, 문자형이면 파싱, 숫자형이면 그대로 사용
    ptsidx = cellfun(@parseGridIndex, pts0(1,:), 'UniformOutput', false);
    ptsidx = cell2mat(ptsidx);  % 2 x N 행렬
    
    % 최종 결과는 각 열(좌표, grid index)을 transpose하여 N x 4 행렬로 구성
    pts = [ptsCoord', ptsidx'];
    
    
    %% 중첩 함수: grid index를 파싱하는 함수
    function res = parseGridIndex(x)
        if ischar(x) || isstring(x)
            % 만약 x가 문자형이면, sscanf로 '[숫자,숫자]' 형태를 파싱
            res = sscanf(char(x), '[%d,%d]', [2,1]);
        elseif isnumeric(x)
            % x가 이미 숫자형이면, 열 벡터로 변환하여 반환
            res = x(:);
        else
            error('grid index의 타입이 예상한 문자 또는 숫자가 아닙니다: %s', class(x));
        end
    end
end
