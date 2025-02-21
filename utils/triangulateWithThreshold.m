function [cgp1, cgp2, filtered_cgpi] = triangulateWithThreshold(gp1, gp2, stereoParams, imgInfo, error_th)
% findGridCorrespondencesWithThreshold - grid 점들(gp1, gp2)로부터 대응점들을 찾고,
%   triangulate를 이용하여 3D 복원 후, reprojection error가 threshold 미만인 점들만 필터링하여 반환합니다.
%
% 입력:
%   gp1, gp2     : grid points; n×4 행렬 (각 행 = [x, y, x index, y index])
%   stereoParams : 스테레오 카메라 파라미터 (stereoParameters 객체)
%   threshold    : reprojection error 임계값 (예: 1)
%
% 출력:
%   filtered_cgp1 : 필터링된 대응점들 (좌측 이미지, p×2)
%   filtered_cgp2 : 필터링된 대응점들 (우측 이미지, p×2)
%   filtered_cgpi : 필터링된 grid index (p×2)

    % 1. 기존 findGridCorrespondences로 대응점 추출
    [cgp1, cgp2, cgpi] = findGridCorrespondences(gp1, gp2);
    
    % 만약 대응점이 없으면 빈 배열 반환
    if isempty(cgp1) || isempty(cgp2)
        cgp1 = [];
        cgp2 = [];
        filtered_cgpi = [];
        return;
    end

    % 2. triangulate를 이용하여 3D 점과 reprojection error 계산
    % matchedPoints는 cgp1, cgp2가 p×2 행렬이어야 하므로 그대로 사용합니다.
    [worldPoints, reprojErrors, ~] = triangulate(cgp1, cgp2, stereoParams);
    
    % 3. reprojection error가 threshold 미만인 점들만 선택
    valid = reprojErrors < error_th;
    
    cgp1 = cgp1(valid, :);
    cgp2 = cgp2(valid, :);
    filtered_cgpi = cgpi(valid, :);
    
    % (선택 사항) reprojection error 벡터도 함께 반환할 수 있습니다.
    % 예: varargout{1} = reprojErrors(valid);

    if isempty(cgp1) || isempty(cgp2)
        warning('Image [%s]: No valid candidate points(re-projection error < %.3g) found in chooseIdx; using findGridCorrespondences instead.', imgInfo, error_th);
        [cgp1, cgp2, ~] = findGridCorrespondences(gp1, gp2);
    end
end
