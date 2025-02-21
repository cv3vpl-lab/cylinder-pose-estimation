function [ln, pts3] = get3dCylinder(gridPts1, gridPts2, stereoParams, cylRadius)
    % makeLineForCylinder
    % 원통의 축선 (ln)과 3D 포인트 (pts3) 계산

    % Stereo Params
    [K1, K2, ~, ~, T_C2_C1, ~] = getCamParams(stereoParams);

    % 그리드 포인트
    gp1 = gridPts1.points;
    gp2 = gridPts2.points;
    gpi1 = gp1(:, 3:4);
    gpi2 = gp2(:, 3:4);
    np1 = size(gpi1, 1);

    % 매칭된 그리드 포인트
    cgp1 = [];
    cgp2 = [];
    gppi = [];
    for i = 1:np1
        idx = find(all(gpi2 == gpi1(i, :), 2));
        if isempty(idx), continue; end

        cgp1 = [cgp1; gp1(i, 1:2)];
        cgp2 = [cgp2; gp2(idx, 1:2)];
        gppi = [gppi; gpi1(i, :)];
    end

    % 3D 포인트
%     [cgp1, cgp2] = chooseIdx(gp1, gp2, stereoParams, 3);

    pts3 = triangulate(cgp1, cgp2, stereoParams)';

    % 실린더 
    [cylinder0, fval] = fitCylinderWPts3(pts3, cylRadius);
%     fprintf('average error : %d mm\n', sqrt(fval));

    % 원통 파라미터 수정
    cylorg0 = cylinder0(1:3)';
    cyldir0 = cylinder0(4:6)';
    if cyldir0(2) < 0
        cyldir0 = -cyldir0;
    end
    [~, i] = min(pts3(2,:));
    cylorg0 = cylorg0 + (cyldir0' * pts3(:, i)) * cyldir0;

    % ln 생성
    ln = [cylorg0, cylorg0 + cyldir0 * 20];
end