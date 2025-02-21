function [pts3, cylT] = fitSingleCylinderV( gridPtsPair, cylRadius, imgPair, fignum, imgInfo, stereoParams )

    [K1, K2, ~, ~, T_C2_C1, ~] = getCamParams(stereoParams);

    % -------------------------------------------------
    % 1) 스테레오 매칭 (chooseIdx) + 3D 점 triangulate
    % -------------------------------------------------
    [cgp1, cgp2] = chooseIdx(gridPtsPair{1}.points, gridPtsPair{2}.points, ...
                             imgInfo, stereoParams, 3, 1);
    pts3  = triangulate( cgp1, cgp2, stereoParams )';  % pts3 : 3×N
    plotReprojectionErrors( cgp1, cgp2, stereoParams );

    % -------------------------------------------------
    % 2) 초기 파라미터를 직접 구해보기
    % -------------------------------------------------
    ctr = mean(pts3, 2);
    coeff = pca(pts3');
    rdir = coeff(:, 3);
    if rdir(3) < 0
        rdir = -rdir;
    end
    
    linePts = [ctr, ctr + rdir];
    d = getDistPts3ToLine( pts3, linePts );
    [~, i] = min(d);
    d2surface = norm(ctr - pts3(:, i));

    [Kcurv, ~] = estCurvatures(pts3);
    cyldir0 = Kcurv(:, 1, i);
    cylorg0 = ctr + rdir * (cylRadius - d2surface);
    cylParams0 = [cylorg0', cyldir0'];

    % (선택) applyCylParamsPrior 적용
    cylParams0 = applyCylParamsPrior(cylParams0, pts3);

    % 초기 축을 동차 변환 행렬 T0로 변환
    T0 = cylParams2T(cylParams0);

    % -------------------------------------------------
    % 4) 최종 실린더 피팅 (기존 fitCylinderWPts3 호출)
    % -------------------------------------------------
    [cylinder0, ~] = fitCylinderWPts3(pts3, cylRadius);
    cylinder0 = applyCylParamsPrior(cylinder0, pts3);
    cylT = cylParams2T( cylinder0 );

    % -------------------------------------------------
    % 3) 초기 축 시각화 (왼쪽 이미지, 연두색)
    % -------------------------------------------------
    figure(fignum(1)), imshow(imgPair{1}), hold on
    figresize;
    drawGridPoints(gridPtsPair{1});
    % 초기 축: 연두색 (예: [0,1,0] for both circle and line)
    initialClrs = [0, 1, 0; 0, 1, 0];  
    drawCylinder( T2vec(T0), cylRadius, K1, initialClrs );

    % 최종 축 시각화 (왼쪽 이미지, 보라색)
    finalClrs = [1, 0, 1; 1, 0, 1];  
    drawCylinder( T2vec(cylT), cylRadius, K1, finalClrs );

    % -------------------------------------------------
    % 5) 오른쪽 이미지에도 표시
    % -------------------------------------------------
    figure(fignum(2)), imshow(imgPair{2}), hold on
    figresize;
    drawGridPoints(gridPtsPair{2});
    % 오른쪽 카메라 좌표계로 변환하여 초기 축과 최종 축 모두 표시
    drawCylinder( T2vec(T_C2_C1 * T0), cylRadius, K2, initialClrs );   % 초기 축 (연두색)
    drawCylinder( T2vec(T_C2_C1 * cylT), cylRadius, K2, finalClrs );     % 최종 축 (보라색)

end
