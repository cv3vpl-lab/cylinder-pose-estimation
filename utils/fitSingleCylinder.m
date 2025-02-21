function [pts3, cylT, fvals, meanError] = fitSingleCylinder(i, gridPtsPair, cylRadius, imgPair, fignum, imgInfo, stereoParams, draw)
    % profile on

    % 카메라 파라미터 추출
    [K1, K2, ~, ~, T_C2_C1, ~] = getCamParams(stereoParams);
    cParam1 = stereoParams.CameraParameters1;
    cParam2 = stereoParams.CameraParameters2;

    % 격자점 선택
%     [cgp1, cgp2, ~] = findGridCorrespondences(gridPtsPair{1}.points, gridPtsPair{2}.points);
%     [cgp1, cgp2, ~] = triangulateWithThreshold(gridPtsPair{1}.points, gridPtsPair{2}.points, stereoParams, imgInfo, 0.5);
    [cgp1, cgp2] = chooseIdx(gridPtsPair{1}.points, gridPtsPair{2}.points, imgInfo, stereoParams, 3, 0.3);

    % 3D 포인트 삼각측량
    [pts3_temp, reprojectionErrors, ~] = triangulate(cgp1, cgp2, stereoParams);
    pts3 = pts3_temp';
    meanError = mean(reprojectionErrors);

    % 실린더 피팅
    [cylinders, fvals] = fitCylinderWPts3(pts3, cylRadius);

    % 실린더 파라미터 보정
    cylinders(1,:) = applyCylParamsPrior(cylinders(1,:), pts3);
    cylinders(2,:) = applyCylParamsPrior(cylinders(2,:), pts3);
    cylT = cylParams2T(cylinders(2,:));

    % 평균 에러 출력
    fprintf('%d-th image [%s]: average error = %.15g -> %.15g mm\n', i, imgInfo, sqrt(fvals(1)), sqrt(fvals(2)));

    %% 시각화 (draw가 true일 때만)
    if draw
        % 재투영 오류 시각화
        plotReprojectionErrors(cgp1, cgp2, stereoParams);

        % 실린더 피팅 시각화
        visualizeCylinderFitting(pts3, cylinders(1,:), cylinders(2,:), cylRadius);

        % 첫 번째 이미지에 그리드와 실린더 표시
        figure(fignum(1)), imshow(imgPair{1}), hold on;
        figresize;
        drawGridPoints(gridPtsPair{1});
        drawCylinder(T2vec(cylT), cylRadius, K1);

        % 두 번째 이미지에 그리드와 실린더 표시
        figure(fignum(2)), imshow(imgPair{2}), hold on;
        figresize;
        drawGridPoints(gridPtsPair{2});
        drawCylinder(T2vec(T_C2_C1 * cylT), cylRadius, K2);
    end

    % profile off
    % profile viewer
end
