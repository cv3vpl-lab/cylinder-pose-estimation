function angle = parseImgInfos(img_infos)
    % parseImgInfos - img_infos 문자열을 숫자 단위로 분리하여 angle 행렬 생성
    %
    % Syntax:
    %   angle = parseImgInfos(img_infos)
    %
    % Inputs:
    %   img_infos - 문자열이 저장된 셀 배열 (예: {'-1-4', '00', '1-8', ...})
    %
    % Outputs:
    %   angle - Nx2 행렬, 각 행은 [num1, num2]
    
    num = length(img_infos);
    angle = zeros(num, 2); % 결과를 저장할 Nx2 행렬 초기화
    
    for i = 1:num
        s = img_infos{i};
        % 정규 표현식을 사용하여 두 개의 숫자 추출 (음수 포함)
        tokens = regexp(s, '^(-?\d+)(-?\d+)$', 'tokens');
        
        if isempty(tokens)
            warning('img_info 형식이 올바르지 않습니다: %s. [0, 0]으로 설정합니다.', s);
            angle(i, :) = [0, 0]; % 기본값 설정 또는 다른 처리
        else
            tokens = tokens{1}; % 첫 번째 매칭 그룹 사용
            num1 = str2double(tokens{1});
            num2 = str2double(tokens{2});
            angle(i, :) = [num1, num2];
        end
    end
end
