function [d, projPts] = getDistPts3ToLine(pts, linePts)
    % pts: 3 x n matrix, linePts: 3 x 2 matrix (두 점으로 직선을 정의)
    
    % 직선을 정의하는 두 점
    p1 = linePts(:, 1);
    p2 = linePts(:, 2);
    
    % 방향 벡터 및 제곱 노름 계산
    v = p2 - p1;
    normv2 = sum(v.^2);
    
    % 각 점에 대해 투영 파라미터(alpha)를 구함
    alphas = sum( (pts - repmat(p1, 1, size(pts,2))) .* repmat(v, 1, size(pts,2)) , 1 ) / normv2;
    
    % 투영된 점 계산
    projPts = repmat(p1, 1, size(pts,2)) + v * alphas;
    
    % 각 점과 투영 점 사이의 거리 계산
    d = sqrt(sum((pts - projPts).^2, 1));
end
