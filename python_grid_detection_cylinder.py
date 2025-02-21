import cv2
import os
from tqdm import tqdm
import json
from utils import util_cylinder
from utils.iotool import undistort_image, load_camera_data

# -----------------------------------
# (1) 메인 함수
# -----------------------------------

def process_images_in_folder(json_path, folder_path, output_folder=None):
    left_camera_params, right_camera_params = load_camera_data(json_path)

    if output_folder is None:
        output_folder = folder_path

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    files = os.listdir(folder_path)
    image_files = [f for f in files if f.lower().endswith(valid_exts)]

    if not image_files:
        print(f"No images found in folder: {folder_path}")
        return

    # Initialize a dictionary to hold all images' JSON data
    images_json_data = {}

    for filename in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(folder_path, filename)
        original_img = cv2.imread(image_path)
    
        if 'L' in filename:
            undistorted_image = undistort_image(original_img, left_camera_params)
        elif 'R' in filename:
            undistorted_image = undistort_image(original_img, right_camera_params)
        else:
            raise ValueError(f"Unknown camera type in filename: {filename}")
            
        # detect_grid: img로부터 grid 찾고, 결과 이미지, 결과 json 반환
        img, result_json, _, _ = detect_grid(undistorted_image)
        base_name = os.path.splitext(filename)[0]
        # Parse JSON data
        try:
            json_data = json.loads(result_json)
            images_json_data[base_name] = json_data
        except json.JSONDecodeError:
            print(f"Invalid JSON data for image {filename}. Skipping.")
            continue
        # Save the processed image
        output_img_name = f"{base_name}_arc{os.path.splitext(filename)[1]}"
        output_img_path = os.path.join(output_folder, output_img_name)
        cv2.imwrite(output_img_path, img)
    
    # Save the collected JSON data into a JSON file
    output_json_path = os.path.join(output_folder, "processed_images_data.json")
    with open(output_json_path, 'w') as json_file:
        json.dump(images_json_data, json_file, indent=4)

    print(f"Data saved to {output_json_path}")
    return json.dumps(images_json_data)       


# @profile
def detect_grid(input_img):
    """
    (원본) 이미지에서 Grid(ARC) 탐지 후, JSON 결과와 결과 이미지를 반환
    """

    try:
        # -----------------------------------
        # 1. 이미지 로드 및 전처리
        # -----------------------------------
        original_img, gray_img, blurred_img, binary_img = util_cylinder.load_and_preprocess_image(input_img)

        # -----------------------------------
        # 2. initial grid points detection
        # -----------------------------------
        horizontal_mask, vertical_mask, centroids = util_cylinder.extract_joints(binary_img)

        # -----------------------------------
        # 3. 실린더 영역 찾기(blob detection => contour => rect)
        # -----------------------------------
        max_contour, mask_contour = util_cylinder.detect_largest_blob(original_img, binary_img, clipLimit=4.5) # time 17.4
        # max_contour = util_cylinder.get_convex_hull(original_img, expansion_pixels=5, visualize=False)

        # -----------------------------------
        # 4. 실린더 영역 내 grid points 필터링 및 중심점(center) 선정
        # -----------------------------------
        final_img, cylinder_centroids, center_point, radius = util_cylinder.find_cylinder_centroids_and_center(
            centroids, max_contour, gray_img, original_img
        )

        # -----------------------------------
        # 5. 마스크 후처리 (ROI 제거, 모폴로지 등)
        # -----------------------------------
        mask_roi_h, mask_roi_v, circle_radius0 = util_cylinder.mask_roi_around_center(
            horizontal_mask, vertical_mask, mask_contour, original_img
        )

        # -----------------------------------
        # 6. grid points 찾기
        # -----------------------------------
        col_img, result_json, rows_updated, cols_updated = util_cylinder.color_and_expand_lines(
            mask_roi_h, mask_roi_v, circle_radius0, center_point, max_contour, mask_contour, original_img, cylinder_centroids
        )
        return col_img, result_json, rows_updated, cols_updated
    except Exception as e:
        print(f"Error in detect_grid: {e}")


if __name__ == "__main__":
    json_path = r'C:\Users\parkchaeho\MATLAB Drive\matlab_0120_pch\delete\data\stereoParams_0117.json'
    # input_path = r'data\input\laser_cylinder'
    input_path = r'data\input\test_cylinder' 
    output_path = r"C:\Users\parkchaeho\MATLAB Drive\matlab_0120_pch\delete\data\output\laser_cylinder_py_0219"
    result_json = process_images_in_folder(json_path, input_path, output_path)

  