function createCameraDataJSON(stereoParams, jsonOutputPath)
    % 두 카메라의 intrinsic 파라미터 추출
    leftCameraParams = stereoParams.CameraParameters1;
    rightCameraParams = stereoParams.CameraParameters2;
    
    % JSON 데이터 생성
    cameraData.LeftCamera = struct('IntrinsicMatrix', leftCameraParams.IntrinsicMatrix', ...
                                    'RadialDistortion', leftCameraParams.RadialDistortion, ...
                                    'TangentialDistortion', leftCameraParams.TangentialDistortion);
    cameraData.RightCamera = struct('IntrinsicMatrix', rightCameraParams.IntrinsicMatrix', ...
                                     'RadialDistortion', rightCameraParams.RadialDistortion, ...
                                     'TangentialDistortion', rightCameraParams.TangentialDistortion);
    
    % JSON 파일 저장
    jsonStr = jsonencode(cameraData);
    fid = fopen(jsonOutputPath, 'w');
    if fid == -1
        error('Could not create JSON file at %s', jsonOutputPath);
    end
    fwrite(fid, jsonStr, 'char');
    fclose(fid);
end