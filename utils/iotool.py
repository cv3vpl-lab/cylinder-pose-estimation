# utils/iotool.py

import json
import cv2
import numpy as np
import os

def load_camera_data(json_path):
    """
    JSON 파일에서 카메라 파라미터를 로드
    
    Args:
        json_path (str): 카메라 파라미터가 저장된 JSON 파일 경로
    
    Returns:
        tuple: 왼쪽(L) 및 오른쪽(R) 카메라 파라미터를 포함한 딕셔너리 반환
    """
    with open(json_path, 'r') as f:
        camera_data = json.load(f)
    return camera_data['LeftCamera'], camera_data['RightCamera']

def undistort_image(image, camera_params):
    """
    단일 이미지에 대해 undistortion을 수행
    
    Args:
        image (numpy.ndarray): 처리할 입력 이미지
        camera_params (dict): 카메라 파라미터 (intrinsic matrix 및 왜곡 계수 포함)
    
    Returns:
        numpy.ndarray: undistortion이 완료된 이미지
    """
    intrinsic_matrix = np.array(camera_params['IntrinsicMatrix'])
    radial_distortion = camera_params['RadialDistortion']
    tangential_distortion = camera_params['TangentialDistortion']
    distortion_coeffs = np.hstack((radial_distortion, tangential_distortion))
    
    undistorted_image = cv2.undistort(image, intrinsic_matrix, distortion_coeffs)
    return undistorted_image

def process_images_in_folder(json_path, input_folder, output_folder):
    """
    폴더 내 모든 이미지에 대해 undistortion을 수행하는 함수
    
    Args:
        json_path (str): 카메라 파라미터가 저장된 JSON 파일 경로
        input_folder (str): 입력 이미지가 저장된 폴더 경로
        output_folder (str): undistortion 결과를 저장할 폴더 경로
    """
    left_camera_params, right_camera_params = load_camera_data(json_path)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    for image_file in os.listdir(input_folder):
        if image_file.endswith('.png'):
            image_path = os.path.join(input_folder, image_file)
            image = cv2.imread(image_path)
            
            # 파일 이름에 따라 왼쪽 또는 오른쪽 카메라 파라미터 사용
            if 'L' in image_file:
                undistorted_image = undistort_image(image, left_camera_params)
            elif 'R' in image_file:
                undistorted_image = undistort_image(image, right_camera_params)
            else:
                print(f"Skipped {image_file}: 파일 이름에 L 또는 R이 포함되어 있지 않음")
                continue
            
            # 결과 이미지 저장
            output_path = os.path.join(output_folder, f"{image_file}")
            cv2.imwrite(output_path, undistorted_image)
            print(f"Processed {image_file} -> Saved to {output_path}")
