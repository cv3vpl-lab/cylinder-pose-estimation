% Add to animateOrbit function
function animateOrbit(views)
    if ~isempty(views)
        rightAxes.Visible = 'on';
        vid = VideoWriter('orbit_recording.avi', 'MPEG-4');
        vid.Quality = 100;
        open(vid);
        
        for i = 1:length(views)
            camPos = views{i}(1:3);
            camTarget = views{i}(4:6);
            camUp = views{i}(7:9);
            
            set(rightAxes, 'CameraPosition', camPos, ...
                         'CameraTarget', camTarget, ...
                         'CameraUpVector', camUp);
            drawnow;
            
            % Capture frame
            F = getframe(rightAxes);
            writeVideo(vid, F);
            pause(0.1);
        end
        close(vid);
        disp('Animation saved as orbit_recording.avi');
    end
end