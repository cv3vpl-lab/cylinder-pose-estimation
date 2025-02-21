function visualize_3D_points(worldPoints)
    % 3D 시각화
    figure;
    scatter3(worldPoints(:, 1), worldPoints(:, 2), worldPoints(:, 3), 'filled');
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    grid on;
    title('3D Reconstruction');
    
    % Reverse the z-axis direction
    ax = gca; % Get current axes
    ax.ZDir = 'reverse'; % Reverse z-axis
end