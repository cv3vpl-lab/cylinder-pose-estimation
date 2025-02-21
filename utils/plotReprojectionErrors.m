function plotReprojectionErrors(matchedPoints1, matchedPoints2, stereoParams)
    % 매트랩 함수 내부에서 변수를 유지하기 위한 persistent 선언
    persistent figNum
    
    % 함수 최초 호출 시 figNum 초기화
    if isempty(figNum)
        figNum = 500;
    else
        figNum = figNum + 1;
    end
    
    % figure(figNum) 호출 -> 500, 501, 502 ... 순으로 열림
    figure(figNum);

    [worldPoints, reprojectionErrors, validIndex] = triangulate(matchedPoints1, matchedPoints2, stereoParams);

    % 유효한 포인트만 선택 (optional)
    validWorldPoints = worldPoints(validIndex, :);
    validErrors = reprojectionErrors(validIndex);
    

    plot(reprojectionErrors, 'bo-', 'LineWidth', 1.5);
    xlabel('포인트 인덱스');
    ylabel('재투영 오차 (픽셀)');
    title(sprintf('전체 데이터: 평균 = %.2f, 분산 = %.2f, 최대 = %.2f, 최소 = %.2f, idx = %d', ...
          mean(reprojectionErrors), var(reprojectionErrors), max(reprojectionErrors), min(reprojectionErrors), length(validIndex)));
    grid on;
end
