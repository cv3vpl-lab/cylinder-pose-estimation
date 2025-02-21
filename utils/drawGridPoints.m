function drawGridPoints(gridPts)
    cp = gridPts.center_point;
    pts = gridPts.points;

    idC = unique(pts(:, 3));
    nC = length(idC);
    
    % 랜덤 컬러 생성 (0~1 사이의 값으로 nC×3 배열)
    clr = rand(nC, 3);

    scatter(cp(1), cp(2), 30, 'r', 'filled');

    for i = 1:nC
        idx = pts(:, 3) == idC(i);
        scatter(pts(idx, 1), pts(idx, 2), 20, clr(i, :), 'filled');
    end
end
