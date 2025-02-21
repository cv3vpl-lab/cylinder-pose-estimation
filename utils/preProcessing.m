function [imgL_uint8, imgR_uint8, imgL_histeq, imgR_histeq] = preProcessing(input_imgL, input_imgR, cameraParams)
    % 좌측 이미지 처리
    imgL_uint8 = im2uint8(input_imgL);
    imgL_uint8 = undistortImage(imgL_uint8, cameraParams.CameraParameters1, 'cubic');
    
    % 컬러 이미지일 경우 그레이스케일로 변환 (이미지가 이미 그레이스케일이면 변환하지 않아도 됨)
    if size(imgL_uint8, 3) == 3
        imgL_uint8 = rgb2gray(imgL_uint8);
    end
    
    imgL_histeq = adapthisteq(imgL_uint8);

    % 우측 이미지 처리
    imgR_uint8 = im2uint8(input_imgR);
    imgR_uint8 = undistortImage(imgR_uint8, cameraParams.CameraParameters2, 'cubic');
    
    if size(imgR_uint8, 3) == 3
        imgR_uint8 = rgb2gray(imgR_uint8);
    end
    
    imgR_histeq = adapthisteq(imgR_uint8);
end
