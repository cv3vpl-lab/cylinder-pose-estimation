function drawFvals(data)
    % drawFvals - 입력 셀 배열 data의 형식에 따라 두 가지 플롯을 그립니다.
    %
    % 만약 data의 각 셀에 1x2 double 배열이 저장되어 있으면,
    %   각 셀의 첫 번째 값과 두 번째 값에 대해 sqrt를 취해 초기/최종 오차를 계산한 후,
    %   이미지 인덱스를 x축으로 하여 두 개의 꺾은선 그래프를 그립니다.
    %
    % 만약 data의 각 셀에 스칼라 값이 저장되어 있으면,
    %   각 셀의 값을 추출하여 이미지 인덱스를 x축으로 꺾은선 그래프로 그립니다.
    
    n = numel(data);
    
    % 각 셀의 크기를 확인합니다.
    sample = data{1};
    if isvector(sample) && numel(sample)==2
        % data의 각 셀은 1x2 배열로 간주 (fvals의 경우)
        init_err = zeros(n,1);
        final_err = zeros(n,1);
        for i = 1:n
            currVals = data{i};  % 예: [102.1082 1.0561]
            init_err(i) = sqrt(currVals(1));
            final_err(i) = sqrt(currVals(2));
        end

        figure;
        plot(1:n, init_err, '-o', 'LineWidth', 2, 'DisplayName', 'Initial Error');
        hold on;
        plot(1:n, final_err, '-s', 'LineWidth', 2, 'DisplayName', 'Final Error');
        xlabel('Image Index');
        ylabel('Reprojection Error (mm)');
        ylim([0 50]);
        title('Reprojection Error Comparison');
        legend('show');
        grid on;
        
    elseif isscalar(sample)
        % data의 각 셀은 스칼라인 경우 (meanError의 경우)
        errors = zeros(n,1);
        for i = 1:n
            errors(i) = data{i};
        end

        figure;
        plot(1:n, errors, '-o', 'LineWidth', 2);
        xlabel('Index');
        ylabel('Mean Error');
        ylim([0 1]);
        title('Mean Reprojection Error');
        grid on;
    else
        error('입력 셀 배열의 각 원소는 1x2 double 배열 또는 스칼라여야 합니다.');
    end
end
