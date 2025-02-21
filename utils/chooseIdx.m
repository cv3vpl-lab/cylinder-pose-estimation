function [cgp1, cgp2] = chooseIdx(gp1, gp2, imgInfo, stereoParams, patchSize, error_th)
% chooseIdx : gp1, gp2에서 patchSize×patchSize 크기의 patch를 1칸씩 슬라이딩하면서
%             각 patch에 대해 triangulate를 이용해 reprojection error를 계산하고,
%             평균 reprojection error가 1픽셀 미만인 patch에 포함된 각 후보 점에 대해
%             기존에 등록된 값과 비교하여 더 낮은 reprojection error를 가진 후보를 선택합니다.
%             최종적으로 선택된 점들을 중복 제거하여 cgp1, cgp2 (p×2 행렬)로 반환합니다.
%
% 입력:
%   gp1, gp2     : M×4 행렬, 각 행 = [x, y, x index, y index]
%   stereoParams : 스테레오 카메라 파라미터 (triangulate 함수 사용)
%   patchSize    : patch의 크기 (n); e.g. 3이면 3×3 patch
%   error_th     : reprojection error threshold e.g. < 0.3
%
% 출력:
%   cgp1, cgp2 : p×2 행렬, 선택된 정상 점들의 좌표 (각각 gp1과 gp2에서 고른 값)

    % 후보 점들을 저장하기 위한 containers.Map 생성
    % key: 'xIndex_yIndex', value: struct('cgp1',[x,y], 'cgp2',[x,y], 'error', reprojError)
    pointMap = containers.Map('KeyType','char','ValueType','any');
    
    % gp1의 grid index (3열, 4열)에서 유일한 x, y 값 추출 (정렬된 순서로)
    unique_x = sort(unique(gp1(:,3)));
    unique_y = sort(unique(gp1(:,4)));
    
    numX = length(unique_x);
    numY = length(unique_y);
    
    % 슬라이딩 윈도우로 가능한 모든 patch 후보 검사
    for ix = 1:(numX - patchSize + 1)
        for iy = 1:(numY - patchSize + 1)
            % 현재 patch에 해당하는 grid index 목록 구성
            candidate_x = unique_x(ix : ix+patchSize-1);
            candidate_y = unique_y(iy : iy+patchSize-1);
            
            % patch 내 모든 grid index 조합 (patchSize^2 × 2 행렬)
            candidateIndices = [];
            for i_x = 1:length(candidate_x)
                for i_y = 1:length(candidate_y)
                    candidateIndices = [candidateIndices; candidate_x(i_x), candidate_y(i_y)];
                end
            end
            
            % gp1, gp2에서 candidateIndices (각 행: [x index, y index])가 모두 존재하는지 확인
            [tf1, loc1] = ismember(candidateIndices, gp1(:,3:4), 'rows');
            [tf2, loc2] = ismember(candidateIndices, gp2(:,3:4), 'rows');
            
            if ~all(tf1) || ~all(tf2)
                continue;  % patch 내에 누락된 점이 있으면 건너뛰기
            end
            
            % 대응하는 후보 점들 추출 (좌표: 1열, 2열)
            candidate_cgp1 = gp1(loc1, 1:2);
            candidate_cgp2 = gp2(loc2, 1:2);
            
            % triangulate를 통해 각 점의 reprojection error 계산
            try
                [~, reprojErrors, ~] = triangulate(candidate_cgp1, candidate_cgp2, stereoParams);
            catch ME
                warning('triangulate 실행 오류 (patch 시작 grid: [%d, %d]): %s', ...
                    candidate_x(1), candidate_y(1), ME.message);
                continue;
            end
            
            % patch의 평균 reprojection error가 1픽셀 미만인 경우에만 정상 patch로 판단
            if mean(reprojErrors) < error_th
                % patch 내 각 점에 대해 처리
                for i_pt = 1:size(candidateIndices, 1)
                    % grid index를 key로 생성 ("xIndex_yIndex")
                    key = sprintf('%d_%d', candidateIndices(i_pt,1), candidateIndices(i_pt,2));
                    currError = reprojErrors(i_pt);
                    curr_cgp1 = candidate_cgp1(i_pt, :);
                    curr_cgp2 = candidate_cgp2(i_pt, :);
                    
                    % 이미 해당 grid index의 후보가 있다면, reprojection error가 낮은 값으로 업데이트
                    if isKey(pointMap, key)
                        stored = pointMap(key);
                        if currError < stored.error
                            pointMap(key) = struct('cgp1', curr_cgp1, 'cgp2', curr_cgp2, 'error', currError);
                        end
                    else
                        pointMap(key) = struct('cgp1', curr_cgp1, 'cgp2', curr_cgp2, 'error', currError);
                    end
                end
            end
        end
    end
    
    % pointMap에 저장된 후보들을 cgp1, cgp2 배열로 추출 (각각 p×2 행렬)
    keysArray = pointMap.keys;
    numPoints = length(keysArray);
    cgp1 = zeros(numPoints, 2);
    cgp2 = zeros(numPoints, 2);
    for i = 1:numPoints
        entry = pointMap(keysArray{i});
        cgp1(i, :) = entry.cgp1;
        cgp2(i, :) = entry.cgp2;
    end
    
    % 만약 cgp1, cgp2가 비어있다면 경고를 출력하고, 
    % gridPtsPair{1}.points, gridPtsPair{2}.points를 사용해 findGridCorrespondences로 대체하여 반환합니다.
    if isempty(cgp1) || isempty(cgp2)
        warning('Image [%s]: No valid candidate points(re-projection error < %.3g) found in chooseIdx; using findGridCorrespondences instead.', imgInfo, error_th);
        [cgp1, cgp2, ~] = findGridCorrespondences(gp1, gp2);
    end
end
