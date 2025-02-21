function [cylParams, fvals] = refindPts(cylParams, pts3, gridPtsPair, stereoParams, cylRadius)
    % 입력:
    % cylParams: 초기 원통 모델 파라미터 (초기화된 값)
    % pts3: 3차원 포인트 (triangulate로 구한 값)
    % gridPtsPair: 그리드 포인트 쌍
    % stereoParams: stere camera parameters
    % cylRadius: 원통 반지름

    % Step 1. 3x3 grid points 대응 관계 찾기
    [cgp1, cgp2] = find3x3GridCorrespondences(gridPtsPair{1}, gridPtsPair{2});

    % Step 2. 해당 grid points를 사용하여 triangulate
    new_pts3 = triangulate(cgp1, cgp2, stereoParams)';

    % Step 3. re-projection error 계산
    errors = computeReprojectionError(new_pts3, cgp1, cgp2, stereoParams);

    % Step 4. error가 1보다 낮은 것만 대응으로 사용
    valid_indices = errors < 1;
    cgp1_valid = cgp1(valid_indices, :);
    cgp2_valid = cgp2(valid_indices, :);

    % Step 5. valid한 포인트로 다시 triangulate하여 새로운 pts3 생성
    pts3_valid = triangulate(cgp1_valid, cgp2_valid, stereoParams)';

    % Step 6. cylinder fitting을 반복적으로 수행
    [cylParams, fvals] = iterativeCylinderFitting(pts3_valid, cylRadius, cylParams);

    % 기존 cylinderParams로 초기화
end


function [cgp1, cgp2] = find3x3GridCorrespondences(gp1, gp2)
    % 3x3 grid points 대응 관계 찾기
    cgp1 = [];
    cgp2 = [];
    
    % gp1과 gp2의 index 정보 추출
    gpi1 = gp1(:,3:4); % x, y index
    gpi2 = gp2(:,3:4);
    
    % 좌상단부터 3x3 block 생성 (index 기준)
    [rows, cols] = getGridSize(gpi1); % 가정: grid size 추출
    
    for i = 1:rows - 2
        for j = 1:cols - 2
            % 3x3 block index 생성 (i,j) ~ (i+2,j+2)
            block_indices = (gpi1(:,1) >= i) & (gpi1(:,1) <= i+2) & ...
                            (gpi1(:,2) >= j) & (gpi1(:,2) <= j+2);
            
            % 해당 block의 grid points 추출
            current_gp1 = gp1(block_indices, :);
            current_gpi1 = current_gp1(:,3:4);
            
            % gp2에서 해당 block의 대응 찾기
            for k = 1:size(current_gp1, 1)
                idx = find(all(gpi2 == current_gpi1(k, :), 2));
                if ~isempty(idx)
                    cgp1 = [cgp1; current_gp1(k, 1:2)];
                    cgp2 = [cgp2; gp2(idx, 1:2)];
                end
            end
        end
    end
    
    % 결과 정렬 (필요 시)
    % [cgp1, cgp2] = sortGridPoints(cgp1, cgp2);
end


function errors = computeReprojectionError(pts3, cgp1, cgp2, stereoParams)
    % pts3: 3D points
    % cgp1: corresponding image points (cam1)
    % cgp2: corresponding image points (cam2)
    % stereoParams: stereo camera parameters
    
    [K1, K2, ~, ~, T_C2_C1, ~] = getCamParams(stereoParams);
    
    % reprojection for cam1
    projected_cgp1 = projectPoints(pts3, K1);
    error1 = sqrt(sum((projected_cgp1 - cgp1).^2, 2));
    
    % reprojection for cam2
    projected_cgp2 = projectPoints(pts3, K2, T_C2_C1);
    error2 = sqrt(sum((projected_cgp2 - cgp2).^2, 2));
    
    % combined error
    errors = error1 + error2;
end

function projected = projectPoints(pts3, cameraMatrix, T)
    % pts3: 3D points (3xN)
    % cameraMatrix: camera intrinsic matrix
    % T: extrinsic transformation (optional)
    
    if nargin == 2
        T = eye(4);
    end
    
    % homogeneous coordinates
    pts3_h = [pts3; ones(1, size(pts3, 2))];
    
    % apply transformation
    pts3_cam = T * pts3_h;
    
    % project to image plane
    projected = cameraMatrix * pts3_cam;
    projected = projected(1:2, :) ./ projected(3, :);
end


function [cylParams, fvals] = iterativeCylinderFitting(pts3, cylRadius, initialCylParams)
    % pts3: valid 3D points after re-projection error filtering
    % cylRadius: cylinder radius
    % initialCylParams: initial cylinder parameters
    
    % 초기화
    cylParams = initialCylParams;
    fvals = [];
    
    % 반복 최적화 (예: 3회)
    for i = 1:3
        % 오차 계산용 함수
        f = @(x) calculateCylinderError(x, pts3, cylRadius);
        
        % fminsearch 사용
        [cylParams, fval] = fminsearch(f, cylParams);
        fvals = [fvals; fval];
    end
end