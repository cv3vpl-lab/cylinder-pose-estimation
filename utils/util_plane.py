import copy
import warnings
import math
import cv2
import re
import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
import random
from skimage import img_as_float
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
import json
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from functools import partial

warnings.simplefilter('ignore', np.RankWarning)

def generate_unique_colors(n, color_list):
    colors = [color_list[i % len(color_list)] for i in range(n)]
    return colors


def label_and_color_masks(mask_roi, color_list):
    """
    이진 마스크에 대해 연결 요소(label) 검출 후, 각 label에 대해 지정 색상으로 컬러링.
    """
    num_labels, labels = cv2.connectedComponents(mask_roi)
    colored_mask = np.zeros((*mask_roi.shape, 3), dtype=np.uint8)
    for label in range(1, num_labels):
        color = color_list[(label - 1) % len(color_list)]
        colored_mask[labels == label] = color
    return num_labels, labels, colored_mask

def get_pca_endpoints(pts):
    """
    pts: (N, 2) float 좌표 집합  
    PCA로 가장 큰 고유값(긴축)의 최소/최대 투영점을 엔드포인트로 반환.
    """
    if len(pts) < 2:
        return (None, None), (None, None)
    mean = np.mean(pts, axis=0)
    centered = pts - mean
    cov = np.cov(centered.T)
    if cov.shape != (2, 2):
        return (None, None), (None, None)
    eigvals, eigvecs = np.linalg.eig(cov)
    max_eig_idx = np.argmax(eigvals)
    principal_axis = eigvecs[:, max_eig_idx]  # (2,)
    projections = np.dot(centered, principal_axis)
    min_idx = np.argmin(projections)
    max_idx = np.argmax(projections)
    x1, y1 = pts[min_idx]
    x2, y2 = pts[max_idx]
    return (x1, y1), (x2, y2)

def create_rotated_line_kernel(size, angle):
    """
    size x size 배열에서 중앙 가로줄(1)을 그린 후 angle만큼 회전시켜
    선형 구조 요소(kernel)을 반환.
    
    예시:
        size=5, angle=-45 인 경우:
            00001
            00010
            00100
            01000
            10000
    """
    kernel = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    cv2.line(kernel, (0, center), (size - 1, center), 1, thickness=1)
    M = cv2.getRotationMatrix2D((center, center), angle, 1)
    rotated = cv2.warpAffine(kernel, M, (size, size), flags=cv2.INTER_NEAREST)
    rotated = (rotated > 0).astype(np.uint8)
    return rotated

def process_contour_info(info, base_mask, patch_size, w, h, global_angle_avg, kernel_size, global_length):
    """
    info: 딕셔너리 {'p1': (x1, y1), 'p2': (x2, y2), 'angle': angle, 'points': pts, 'length': length}
    base_mask: 이진화된 입력 mask (객체: 255, 배경: 0)
    patch_size: 엔드포인트 주변 추출할 영역 크기
    w, h: base_mask의 너비, 높이
    global_angle_avg: 모든 컨투어의 각도 중앙값
    kernel_size: dilation에 사용할 커널 크기
    global_length: 모든 컨투어 중 PCA로 구한 가장 긴 축의 최대 길이
    
    → 각 컨투어의 두 엔드포인트 주변 patch(expand_roi)를 추출하고,  
       abnormal한 경우(global_angle_avg 사용) 또는 원래 angle로 회전 커널을 생성하여
       dilation을 수행한 결과(확장 마스크와 디버그 이미지를)를 반환합니다.
       
       단, (이제 PCA 길이에 따른 확장 중단 조건은 제거되어 항상 확장 작업을 수행합니다.)
       
    반환:
        (expanded_contribution, debug_contribution)
            expanded_contribution: 해당 컨투어의 확장 결과 (base_mask와 동일 크기의 binary mask)
            debug_contribution: 해당 컨투어의 확장 영역을 연두색(BGR: (144,238,144))으로 표시한 이미지 (3채널)
    """
    expanded_contribution = np.zeros_like(base_mask)
    debug_contribution = np.zeros((h, w, 3), dtype=np.uint8)
    p1, p2, angle = info['p1'], info['p2'], info['angle']
    comp_length = info.get('length', None)

    # angle, p1, p2가 None인 경우에는 확장하지 않음.
    if angle is None or p1 is None or p2 is None or (comp_length is not None and comp_length > 0.8 * global_length):
        return expanded_contribution, debug_contribution

    # 만약 개별 angle이 global_angle_avg와 5도 이상 차이가 나면 abnormal로 판단 → global_angle_avg 사용
    if abs(angle - global_angle_avg) > 5.0:
        angle_kernel = global_angle_avg
    else:
        angle_kernel = angle

    # 회전된 선형 구조 요소(커널) 생성
    kernel = create_rotated_line_kernel(kernel_size, angle_kernel)
    half = patch_size // 2

    for endpoint in [p1, p2]:
        cx, cy = int(round(endpoint[0])), int(round(endpoint[1]))
        endpoint_mask = np.zeros_like(base_mask)
        x1_ = max(cx - half, 0)
        x2_ = min(cx + half + 1, w)
        y1_ = max(cy - half, 0)
        y2_ = min(cy + half + 1, h)
        patch = base_mask[y1_:y2_, x1_:x2_]
        endpoint_mask[y1_:y2_, x1_:x2_] = patch
        # dilation 적용 후 erosion (노이즈 보정)
        dilated = cv2.dilate(endpoint_mask, kernel, iterations=1)
        dilated = cv2.erode(dilated, np.ones((3, 3), np.uint8), iterations=1)
        expanded_contribution = cv2.bitwise_or(expanded_contribution, dilated)
        # 디버그: 해당 영역을 연두색(BGR: (144,238,144))으로 표시
        sub_debug = debug_contribution[y1_:y2_, x1_:x2_]
        sub_mask = dilated[y1_:y2_, x1_:x2_] > 0
        sub_debug[sub_mask] = (144, 238, 144)
        debug_contribution[y1_:y2_, x1_:x2_] = sub_debug

    return expanded_contribution, debug_contribution


def expand_line_roi(mask_roi, patch_size=15, kernel_size=81, min_pixels=8, max_pixels=700):
    """
    1) 모든 컨투어에 대해 PCA 각도(angle)와 PCA 길이(length)를 구하여,  
       전체 컨투어의 중앙 각도(global_angle_avg)와 가장 긴 축의 최대 길이(global_length)를 계산합니다.
    2) 각 컨투어별로 엔드포인트 주변 patch 영역(expand_roi)을 추출한 후,  
       abnormal한 경우(global_angle_avg 사용) 또는 원래 angle로 회전 커널을 생성하여 dilation을 수행합니다.
       단, 해당 컨포넌트의 PCA 길이가 0.8×global_length보다 큰 경우에는 확장하지 않습니다.
       
    반환:
        combined_mask: 원본 mask와 확장 영역을 합한 binary mask
        debug_img: 확장 영역을 연두색으로 표시한 디버깅용 이미지 (BGR)
    """
    # morphology로 노이즈 제거
    mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    # 이진화: 객체=255, 배경=0
    base_mask = np.where(mask_roi > 0, 255, 0).astype(np.uint8)
    h, w = base_mask.shape

    # 디버그용 이미지 (BGR)
    debug_img = cv2.cvtColor(base_mask, cv2.COLOR_GRAY2BGR)
    # 결과 누적용 mask
    expanded_mask = base_mask.copy()

    # 컨투어 추출
    contours, _ = cv2.findContours(base_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # (1) 모든 컨투어에 대해 PCA 각도(angle)와 길이(length)를 구하여 info 저장
    angle_list = []
    length_list = []
    expansions_info = []  # 각 원소: {'p1':..., 'p2':..., 'angle':..., 'points':..., 'length':...}
    for cnt in contours:
        pts = cnt.reshape(-1, 2).astype(np.float32)
        if len(pts) < min_pixels or len(pts) > max_pixels:
            expansions_info.append({'p1': None, 'p2': None, 'angle': None, 'points': pts, 'length': None})
            continue
        (x1, y1), (x2, y2) = get_pca_endpoints(pts)
        if x1 is None or x2 is None:
            expansions_info.append({'p1': None, 'p2': None, 'angle': None, 'points': pts, 'length': None})
            continue
        dx, dy = (x2 - x1), (y2 - y1)
        length = np.hypot(dx, dy)
        if length < 1e-8:
            expansions_info.append({'p1': None, 'p2': None, 'angle': None, 'points': pts, 'length': None})
            continue
        angle = -np.degrees(np.arctan2(dy, dx))
        angle_list.append(angle)
        length_list.append(length)
        expansions_info.append({'p1': (x1, y1), 'p2': (x2, y2), 'angle': angle, 'points': pts, 'length': length})

    if len(angle_list) == 0 or len(length_list) == 0:
        # 확장할 컨투어가 없으면 원본 반환
        return base_mask, debug_img

    # global_angle_avg: 모든 컨투어의 각도의 중앙값
    global_angle_avg = np.median(angle_list)
    # global_length: 모든 컨투어 중 가장 긴 PCA 축의 최대 길이
    global_length = max(length_list)

    # (2) 각 컨투어별 확장 작업을 ThreadPoolExecutor로 병렬 처리
    func = partial(process_contour_info, 
                   base_mask=base_mask, 
                   patch_size=patch_size, 
                   w=w, h=h, 
                   global_angle_avg=global_angle_avg, 
                   kernel_size=kernel_size,
                   global_length=global_length)
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(func, expansions_info))
    # results: list of tuple (expanded_contribution, debug_contribution)
    for exp_cont, dbg_cont in results:
        expanded_mask = cv2.bitwise_or(expanded_mask, exp_cont)
        debug_img = cv2.bitwise_or(debug_img, dbg_cont)

    # 최종 결합: 원본과 확장 mask
    combined_mask = cv2.bitwise_or(expanded_mask, base_mask)
    return combined_mask, debug_img

def expands_line_roi(horizontal_expanded, num, mask_contour, patch_size=15, kernel_size=81):
    """
    여러 번 반복하여 확장을 수행하는 함수.
    각 반복마다 expand_line_roi를 호출하여 확장을 적용한 후,
    mask_contour와 bitwise 연산을 수행한다.
    
    Parameters:
        horizontal_expanded: 입력 마스크
        num: 반복 횟수
        mask_contour: 추가적으로 결합할 마스크 (예: 원래의 ROI contour)
        patch_size: 엔드포인트 주변 추출 영역 크기
        kernel_size: dilation에 사용할 커널 크기
        
    Returns:
        horizontal_expanded: 최종 확장된 binary 마스크
        debug_img: 마지막 반복의 디버깅용 이미지 (BGR)
    """
    debug_img = None
    for i in range(num):
        horizontal_expanded, debug_img = expand_line_roi(horizontal_expanded,
                                                         patch_size=patch_size,
                                                         kernel_size=kernel_size)
        horizontal_expanded = cv2.bitwise_and(horizontal_expanded, mask_contour)
    return horizontal_expanded, debug_img



def visualize_centroids_with_roi(input_img, cylinder_centroids, roi_mask,
                                 circle_radius=2, circle_color=(255, 0, 0),
                                 roi_color=(0, 255, 0), alpha=0.3):
    """
    입력 이미지 위에 cylinder_centroids의 포인트들을 원으로 그리며,
    ROI를 반투명한 초록색으로 오버레이하여 시각화하는 함수.

    Parameters:
        input_img (numpy.ndarray): 원본 컬러 이미지.
        cylinder_centroids (list of tuples): 그릴 점들의 리스트 [(cX1, cY1), (cX2, cY2), ...].
        roi_mask (numpy.ndarray): ROI를 나타내는 이진 마스크 (단일 채널, 값은 0 또는 255).
        circle_radius (int): 그릴 원의 반경 (기본값: 2).
        circle_color (tuple): 원의 색상 (BGR 형식, 기본값: 빨간색 (255, 0, 0)).
        roi_color (tuple): ROI 오버레이의 색상 (BGR 형식, 기본값: 초록색 (0, 255, 0)).
        alpha (float): ROI 오버레이의 투명도 (0.0 ~ 1.0, 기본값: 0.3).

    Returns:
        output_img (numpy.ndarray): 시각화된 이미지.
    """
    # 1. 입력 이미지가 컬러인지 확인
    if len(input_img.shape) == 2 or input_img.shape[2] == 1:
        input_img_color = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)
    else:
        input_img_color = input_img.copy()

    # 2. roi_mask의 데이터 타입 확인 및 변환
    # print("roi_mask dtype before conversion:", roi_mask.dtype)
    if roi_mask.dtype != np.uint8:
        # 값이 0과 1 사이에 있다면 0과 255로 스케일링
        if roi_mask.max() <= 1:
            roi_mask = (roi_mask * 255).astype(np.uint8)
        else:
            # 다른 스케일의 값이라면 0과 255로 클리핑 후 변환
            roi_mask = cv2.normalize(roi_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # print("roi_mask dtype after conversion:", roi_mask.dtype)

    # 3. ROI 오버레이 생성
    # ROI 마스크를 컬러로 변환
    if len(roi_mask.shape) == 2:
        roi_mask_color = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR)
    else:
        roi_mask_color = roi_mask.copy()

    # 초록색 오버레이 생성
    green_overlay = np.zeros_like(input_img_color, dtype=np.uint8)
    green_overlay[:] = roi_color  # 전체 이미지를 초록색으로 채움

    # ROI 영역만 초록색으로 설정
    green_masked = cv2.bitwise_and(green_overlay, roi_mask_color)

    # 원본 이미지와 오버레이 블렌딩
    output_img = cv2.addWeighted(input_img_color, 1, green_masked, alpha, 0)

    # 3. X 모양 마커 오버레이 생성
    marker_overlay = np.zeros_like(output_img)
    for (cX, cY) in cylinder_centroids:
        # 대각선 2개로 X 모양 구성
        cv2.line(marker_overlay, 
                 (cX - circle_radius, cY - circle_radius),
                 (cX + circle_radius, cY + circle_radius),
                 circle_color, 1)
        cv2.line(marker_overlay,
                 (cX + circle_radius, cY - circle_radius),
                 (cX - circle_radius, cY + circle_radius),
                 circle_color, 1)

    # 4. 마커 오버레이에 0.5 투명도 적용
    output_img = cv2.addWeighted(output_img, 1.0, marker_overlay, 0.5, 0)

    return output_img

def group_points_by_label(points, labels, x_offset, y_offset):
    points_grouped = {}
    for point in points:
        cX, cY = point
        rx = cX - x_offset
        ry = cY - y_offset
        if 0 <= ry < labels.shape[0] and 0 <= rx < labels.shape[1]:
            label = labels[ry, rx]
            if label > 0:
                if label not in points_grouped:
                    points_grouped[label] = []
                points_grouped[label].append((cX, cY))
    points_grouped = sort_rows(points_grouped)
    return points_grouped

def create_dummy_rows_cols(sorted_rows, sorted_cols, degree=2):
    """
    정렬된 row와 col 데이터를 받아서, 각 row와 col에 대해 더미 초기화된 equations를 추가하여
    딕셔너리 형태로 반환하는 함수.
    
    Args:
        sorted_rows (list of tuple): 각 튜플은 (label, points) 형태이며, row 데이터.
        sorted_cols (list of tuple): 각 튜플은 (label, points) 형태이며, col 데이터.
        degree (int): 다항식의 차수. 각 equation은 [0]*(degree+4) 길이의 리스트로 초기화됨.
    
    Returns:
        tuple: (rows, cols) 딕셔너리.
               rows = {"points": {"row1": points, ...},
                       "equations": {"row1": [0]*(degree+4), ...}}
               cols = {"points": {"col1": points, ...},
                       "equations": {"col1": [0]*(degree+4), ...}}
    """
    rows = {"points": {}, "equations": {}}
    for i, (label, points) in enumerate(sorted_rows, start=1):
        row_name = f"row{i}"
        rows["points"][row_name] = points
        rows["equations"][row_name] = [0] * (degree + 4)
    
    cols = {"points": {}, "equations": {}}
    for i, (label, points) in enumerate(sorted_cols, start=1):
        col_name = f"col{i}"
        cols["points"][col_name] = points
        cols["equations"][col_name] = [0] * (degree + 4)
    
    return rows, cols

# Function to sort rows by y-coordinate
def sort_rows(points_grouped):
    sorted_rows = sorted(points_grouped.items(), key=lambda item: min(point[1] for point in item[1]))
    return sorted_rows

# Function to sort columns by x-coordinate
def sort_cols(points_grouped):
    sorted_cols = sorted(points_grouped.items(), key=lambda item: min(point[0] for point in item[1]))
    return sorted_cols

def polynomial_fitting(centers, degree=3):
    """
    다항식 피팅을 사용해 중심선을 매끄럽게 만듭니다.
    :param centers: (N x 2) 형태의 numpy array, 각 행은 (x, y) 좌표
    :param degree: 다항식 차수 (기본값: 3)
    :return: (x_fitted, y_fitted)
    """
    # x, y 좌표 분리
    x = centers[:, 0]
    y = centers[:, 1]

    # 다항식 피팅
    poly_coeff = np.polyfit(x, y, degree)
    poly_func = np.poly1d(poly_coeff)

    # 피팅된 좌표 계산
    x_fitted = np.linspace(np.min(x), np.max(x), len(x))
    y_fitted = poly_func(x_fitted)

    return x_fitted, y_fitted

def polynomial_fitting_row(pts_x, pts_y, degree=3):
    """
    행(row)에 대한 3차 다항식 y=f(x) 피팅
    """
    poly_coeff = np.polyfit(pts_x, pts_y, degree)
    # poly_coeff = [a3, a2, a1, a0],  highest power first
    # y = a3*x^3 + a2*x^2 + a1*x + a0
    return poly_coeff

def polynomial_fitting_col(pts_y, pts_x, degree=3):
    """
    열(col)에 대한 3차 다항식 x=f(y) 피팅
    """
    poly_coeff = np.polyfit(pts_y, pts_x, degree)
    # poly_coeff = [b3, b2, b1, b0]
    # x = b3*y^3 + b2*y^2 + b1*y + b0
    return poly_coeff


def fit_and_draw_polynomial(img, rows, cols, max_w, max_h, max_contour, degree=3):
    """
    행(row): y=f(x)
    열(col): x=f(y)
    각각 n차 다항식으로 피팅해 그림 + 'equations'에 저장
    추가로 비정상적인 열을 병합하여 새로운 다항식을 피팅하고 그림을 업데이트
    디버깅을 위해 로그 추가
    """
    img_with_poly = img.copy()

    # (B) 열에 대한 처리: 비정상적인 열 병합 로직 추가
    # 1. 열에 대한 다항식 피팅
    for col_name, points in cols["points"].items():
        if len(points) < degree + 1:
            # print(f"[DEBUG] Column '{col_name}'의 점 개수가 부족하여 스킵됩니다. (점 개수: {len(points)})")
            continue  # 피팅을 위해 최소 degree+1개의 점 필요

        pts = np.array(points, dtype=np.float32)
        # y 기준 정렬
        pts = pts[np.argsort(pts[:, 1])]

        y_vals = pts[:, 1]
        x_vals = pts[:, 0]

        # 다항식 피팅
        poly_coeff = polynomial_fitting_col(y_vals, x_vals, degree=degree)
        # 식: x = poly_coeff[0]*y^n + poly_coeff[1]*y^(n-1) + ... + poly_coeff[n]*y + poly_coeff[n+1]

        # 도메인 설정
        y_min, y_max = y_vals.min(), y_vals.max()
        y_min = y_min - 10
        y_max = y_max + 10

        # equations에 저장
        cols["equations"][col_name] = list(poly_coeff) + [float(y_min), float(y_max), abs(float(y_min)-float(y_max))]

        # print(f"[DEBUG] Column '{col_name}' 피팅 완료: 계수={poly_coeff}, y_min={y_min}, y_max={y_max}, abs_diff={abs(y_min - y_max)}")

    # 2. 변수 정의: 모든 열의 abs(min - max) 중 최댓값
    if cols["equations"]:
        threshold_value = max(
            abs(equation[-1]) for equation in cols["equations"].values()
        )
    else:
        threshold_value = 0
    # print(f"[DEBUG] Threshold Value (최대 abs(min - max)): {threshold_value}")

    # 3. 비정상적인 열 식별: abs(min - max) <= 0.9 * threshold_value
    abnormal_cols = [
        col_name for col_name, equation in cols["equations"].items()
        if abs(equation[-1]) <= 0.9 * threshold_value
    ]
    # print(f"[DEBUG] 비정상적인 열 목록: {abnormal_cols}")

    # 각 열의 abs(min - max) 값 로그 출력
    for col_name, equation in cols["equations"].items():
        abs_diff = abs(equation[-1])
        # print(f"[DEBUG] 열 '{col_name}': abs(min - max) = {abs_diff}")

    # 4. 병합 그룹 형성: 연속된 비정상 열 중 누적 abs(min - max)가 threshold_value를 넘지 않도록 그룹화
    merge_groups = []
    current_group = []
    cumulative = 0

    # 열 이름을 숫자 순으로 정렬 (col1, col2, ..., colN)
    sorted_col_names = sorted(cols["equations"].keys(), key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))

    for col_name in sorted_col_names:
        if col_name in abnormal_cols:
            col_diff = abs(cols["equations"][col_name][-1])
            # print(f"[DEBUG] 현재 열 '{col_name}'의 abs_diff: {col_diff}, 현재 누적값: {cumulative}")
            if cumulative + col_diff <= threshold_value:
                current_group.append(col_name)
                cumulative += col_diff
                # print(f"[DEBUG] '{col_name}'을 현재 그룹에 추가. 누적값: {cumulative}")
            else:
                if current_group:
                    merge_groups.append(current_group)
                    # print(f"[DEBUG] 현재 그룹 {current_group}을 merge_groups에 추가.")
                current_group = [col_name]
                cumulative = col_diff
                # print(f"[DEBUG] 새로운 그룹 시작: {current_group}, 누적값: {cumulative}")
        else:
            if current_group:
                merge_groups.append(current_group)
                # print(f"[DEBUG] 현재 그룹 {current_group}을 merge_groups에 추가.")
                current_group = []
                cumulative = 0

    if current_group:
        merge_groups.append(current_group)
        # print(f"[DEBUG] 마지막 그룹 {current_group}을 merge_groups에 추가.")

    # print(f"[DEBUG] 최종 병합 그룹: {merge_groups}")

    # 5. 병합 그룹을 이용해 새로운 다항식 피팅 및 equations, points 업데이트
    for group in merge_groups:
        merged_points = []
        for col_name in group:
            merged_points.extend(cols["points"][col_name])
            # print(f"[DEBUG] 그룹 '{group}'에 속한 열 '{col_name}'의 점을 병합.")
            # 기존 열 삭제
            del cols["points"][col_name]
            del cols["equations"][col_name]
            # print(f"[DEBUG] 열 '{col_name}'을 삭제했습니다.")
        
        if len(merged_points) < degree + 1:
            # print(f"[DEBUG] 병합된 그룹 '{group}'의 점 개수가 부족하여 스킵됩니다. (점 개수: {len(merged_points)})")
            continue  # 피팅을 위해 최소 degree+1개의 점 필요

        pts = np.array(merged_points, dtype=np.float32)
        # y 기준 정렬
        pts = pts[np.argsort(pts[:, 1])]

        y_vals = pts[:, 1]
        x_vals = pts[:, 0]

        # 새로운 다항식 피팅
        poly_coeff = polynomial_fitting_col(y_vals, x_vals, degree=degree)

        # 도메인 설정
        y_min, y_max = y_vals.min(), y_vals.max()

        # equations에 새로운 병합 열 저장
        merged_col_name = "_".join(group)  # 병합된 열 이름 예: "col2_col3"
        cols["equations"][merged_col_name] = list(poly_coeff) + [float(y_min), float(y_max), abs(float(y_min)-float(y_max))]
        # print(f"[DEBUG] 병합 그룹 '{group}'을 '{merged_col_name}'으로 저장. 계수={poly_coeff}, y_min={y_min}, y_max={y_max}, abs_diff={abs(y_min - y_max)}")

        # points에 병합된 점 저장
        cols["points"][merged_col_name] = merged_points
        # print(f"[DEBUG] 병합된 열 '{merged_col_name}'에 점을 저장했습니다.")

    # 5a. 병합 후 열 라벨을 재정렬하고 연속적인 이름으로 변경
    # 라벨을 정렬한 후 새로운 라벨을 'col1', 'col2', ...로 재정의
    sorted_labels = sorted(cols["equations"].keys(), key=lambda x: int(x.split('_')[0].replace('col','')))

    new_equations = {}
    new_points = {}

    for idx, label in enumerate(sorted_labels, start=1):
        new_label = f'col{idx}'
        new_equations[new_label] = cols["equations"][label]
        new_points[new_label] = cols["points"][label]
        # print(f"[DEBUG] 열 '{label}'을 '{new_label}'으로 재명명했습니다.")

    cols["equations"] = new_equations
    cols["points"] = new_points

    # 6. 병합 후 모든 열(병합된 열과 비병합된 정상 열)을 그리기
    for col_name, equation in cols["equations"].items():
        if len(cols["points"][col_name]) < degree + 1:
            # print(f"[DEBUG] 열 '{col_name}'의 점 개수가 부족하여 스킵됩니다. (점 개수: {len(cols['points'][col_name])})")
            continue  # 피팅을 위해 최소 degree+1개의 점 필요

        pts = np.array(cols["points"][col_name], dtype=np.float32)
        # y 기준 정렬
        pts = pts[np.argsort(pts[:, 1])]

        y_vals = pts[:, 1]
        x_vals = pts[:, 0]

        # 다항식 계수
        poly_coeff = equation[:degree + 1]

        # 도메인 설정
        y_min, y_max = y_vals.min(), y_vals.max()
        y_min = y_min - 50
        y_max = y_max + 50
        cols["equations"][col_name] = list(poly_coeff) + [float(y_min), float(y_max), abs(float(y_min)-float(y_max))]


        # print(f"[DEBUG] 열 '{col_name}' 피팅 완료: 계수={poly_coeff}, y_min={y_min}, y_max={y_max}, abs_diff={abs(y_min - y_max)}")

        # 곡선 샘플링
        num_points = max(50, len(y_vals))
        y_samp = np.linspace(y_min, y_max, num_points)
        # 다항식 값 계산
        x_samp = np.polyval(poly_coeff, y_samp)

        # 이미지에 곡선 그리기
        for i in range(num_points - 1):
            pt1 = (int(x_samp[i]), int(y_samp[i]))
            pt2 = (int(x_samp[i+1]), int(y_samp[i+1]))
            cv2.line(img_with_poly, pt1, pt2, (255, 0, 0), 1)  # 파란색으로 그리기

    # (A) 행에 대한 처리: 기존 로직 유지
    for row_name, points in rows["points"].items():
        if len(points) < degree + 1:
            # print(f"[DEBUG] Row '{row_name}'의 점 개수가 부족하여 스킵됩니다. (점 개수: {len(points)})")
            continue  # 피팅을 위해 최소 degree+1개의 점 필요

        pts = np.array(points, dtype=np.float32)
        # x 기준 정렬
        pts = pts[np.argsort(pts[:, 0])]

        x_vals = pts[:, 0]
        y_vals = pts[:, 1]

        # 다항식 피팅
        poly_coeff = polynomial_fitting_row(x_vals, y_vals, degree=degree)
        # 식: y = poly_coeff[0]*x^n + poly_coeff[1]*x^(n-1) + ... + poly_coeff[n]*x + poly_coeff[n+1]

        # 도메인 설정
        x_min, x_max = x_vals.min(), x_vals.max()
        x_min = x_min - 50
        x_max = x_max + 50
        # equations에 저장
        rows["equations"][row_name] = list(poly_coeff) + [float(x_min), float(x_max), abs(float(x_min)-float(x_max))]

        # print(f"[DEBUG] Row '{row_name}' 피팅 완료: 계수={poly_coeff}, x_min={x_min}, x_max={x_max}, abs_diff={abs(x_min - x_max)}")

        # 곡선 샘플링
        num_points = max(50, len(x_vals))
        x_samp = np.linspace(x_min, x_max, num_points)
        # 다항식 값 계산
        y_samp = np.polyval(poly_coeff, x_samp)

        # 이미지에 곡선 그리기
        for i in range(num_points - 1):
            pt1 = (int(x_samp[i]), int(y_samp[i]))
            pt2 = (int(x_samp[i+1]), int(y_samp[i+1]))
            cv2.line(img_with_poly, pt1, pt2, (0, 255, 0), 1)  # 녹색으로 그리기

    return img_with_poly, rows, cols

def remove_label(rows, cols):
    """
    Removes labels from rows and cols where the remaining length after considering
    n_start and n_end is invalid.

    Args:
        rows (dict): Dictionary with row labels as keys and equations as values.
        cols (dict): Dictionary with column labels as keys and equations as values.

    Returns:
        tuple: Updated (rows, cols) dictionaries after removing labels with invalid equation lengths.
    """
    def remove_labels(data, n_start, n_end, prefix='col'):
        """
        data 딕셔너리는 'equations'와 'points' 키를 가지고 있으며,
        각 키의 값은 레이블(예: 'col1', 'col2', ...)을 키로 하는 딕셔너리입니다.
        
        1. 전체 레이블 목록에서 앞의 n_start개와 뒤의 n_end개에 해당하는 레이블을 삭제합니다.
        2. 남은 레이블들을 원래 순서대로 "prefix1", "prefix2", ... 형식으로 재네이밍합니다.
        
        예를 들어, 기존 레이블이 col1, col2, col3, col4, col5 였을 때
        n_start=1, n_end=1이면 col1과 col5가 삭제되고,
        남은 col2, col3, col4가 col1, col2, col3으로 다시 이름이 바뀝니다.
        """
        # 삭제 전에 원래 키 순서를 저장 (order 유지)
        original_keys = list(data['equations'].keys())
        
        # 삭제할 키: 앞 n_start개와 뒤 n_end개
        keys_to_remove = original_keys[:n_start] + (original_keys[-n_end:] if n_end > 0 else [])
        
        # 지정된 키들을 'equations'와 'points'에서 삭제(del)
        for key in keys_to_remove:
            if key in data['equations']:
                del data['equations'][key]
            if key in data['points']:
                del data['points'][key]
        
        # 남은 키들을 원래 순서대로 결정하기 위해 original_keys에서 삭제된 키들을 제외
        remaining_keys = [key for key in original_keys if key not in keys_to_remove]
        
        # 새로운 딕셔너리 생성 (재네이밍)
        new_equations = {}
        new_points = {}
        for idx, old_key in enumerate(remaining_keys, start=1):
            new_key = f"{prefix}{idx}"
            new_equations[new_key] = data['equations'][old_key]
            new_points[new_key] = data['points'][old_key]
        
        data['equations'] = new_equations
        data['points'] = new_points
        return data

    # Remove labels from rows
    rows = remove_labels(rows, n_start=1, n_end=0)
    
    # Remove labels from cols
    cols = remove_labels(cols, n_start=0, n_end=1)

    return rows, cols

def compute_center_of_gravity_y(gray_img, points, window_size=5):
    """
    주어진 점들(points)에서 x좌표는 고정, y좌표만 그레이스케일 중력중심으로 보정.
    points는 (x, y) 튜플의 리스트 형태.
    """
    refined = []
    half = window_size // 2
    H, W = gray_img.shape[:2]

    for x, y in points:
        ix = int(round(x))
        # y는 float 유지
        iy = y  
        
        # 윈도우 범위 계산
        top = max(int(np.floor(iy)) - half, 0)
        bottom = min(int(np.ceil(iy)) + half + 1, H)

        # 범위가 유효한지 확인
        if ix < 0 or ix >= W:
            # 이미지 밖이면 보정 없이 추가
            refined.append((x, y))
            continue

        # 윈도우 픽셀값
        roi = gray_img[top:bottom, ix]
        y_indices = np.arange(top, bottom)

        G = roi.astype(np.float32)
        s = np.sum(G)
        if s == 0:
            refined.append((x, y))
            continue

        # y중심 = Σ(y_i * 픽셀값) / Σ(픽셀값)
        cog_y = np.sum(y_indices * G) / s
        # 이동량 제한(±0.5 예시)
        delta = cog_y - y
        if abs(delta) > 0.5:
            delta = 0.5 if delta > 0 else -0.5

        new_y = y + delta
        new_y = np.clip(new_y, 0, H - 1)
        refined.append((x, new_y))

    return np.array(refined, dtype=np.float32)


def compute_center_of_gravity_x(gray_img, points, window_size=5):
    """
    주어진 점들(points)에서 y좌표는 고정, x좌표만 그레이스케일 중력중심으로 보정.
    points는 (x, y) 튜플의 리스트 형태.
    """
    refined = []
    half = window_size // 2
    H, W = gray_img.shape[:2]

    for x, y in points:
        iy = int(round(y))
        ix = x

        # 윈도우 범위
        left = max(int(np.floor(ix)) - half, 0)
        right = min(int(np.ceil(ix)) + half + 1, W)

        if iy < 0 or iy >= H:
            refined.append((x, y))
            continue

        # 윈도우 픽셀값 (행: iy 고정, 열: left~right)
        roi = gray_img[iy, left:right]
        x_indices = np.arange(left, right)

        G = roi.astype(np.float32)
        s = np.sum(G)
        if s == 0:
            refined.append((x, y))
            continue

        # x중심
        cog_x = np.sum(x_indices * G) / s
        # 이동량 제한(±0.5 예시)
        delta = cog_x - x
        if abs(delta) > 0.5:
            delta = 0.5 if delta > 0 else -0.5

        new_x = x + delta
        new_x = np.clip(new_x, 0, W - 1)
        refined.append((new_x, y))

    return np.array(refined, dtype=np.float32)


def modify_grayscale_Cline(
    input_img, rows, cols, 
    draw_points=True, 
    degree=1,
    sample_step=0.5,  # 0.5 픽셀 단위로 변경
    window_size=5
):
    """
    1) rows["equations"], cols["equations"]에 저장된 eq로부터 점들을 샘플링
    2) 행(row)은 y좌표만, 열(col)은 x좌표만 그레이스케일 중력중심 보정
    3) 보정된 점들로 다시 degree차 다항식 피팅
    4) eq 갱신. draw_points=True면, 보정 후 라인을 input_img 위에 그려 output_img 반환.

    * 1차(직선) 기준: row eq = [a, b, x_min, x_max, ...], col eq = [a, b, y_min, y_max, ...]
    """
    # 1. 입력 이미지가 컬러인지 여부 확인 -> 그레이스케일 변환
    if len(input_img.shape) == 2:
        gray_img = input_img.copy()
        out_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR) if draw_points else None
    else:
        gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        out_img = input_img.copy() if draw_points else None

    # float 변환(중력중심)
    gray_float = img_as_float(gray_img)

    rows_updated = copy.deepcopy(rows)
    cols_updated = copy.deepcopy(cols)

    # === (A) Rows 갱신 ===
    for label, eq in rows_updated.get("equations", {}).items():
        # eq 길이 체크 (a, b, x_min, x_max, ...)
        if len(eq) < degree + 3:
            # 1차면 최소 5개 (a, b, xMin, xMax, dummy)
            continue

        # 1차 방정식만 처리
        if degree == 1:
            # eq = [a, b, xMin, xMax, dummy]
            poly_coeff = eq[:degree+1]  # [a, b]
            x_min = eq[degree+1]
            x_max = eq[degree+2]
            if x_max < x_min:
                continue

            # 1) (x_min ~ x_max) 0.5 픽셀 단위로 샘플링
            sampled_points = []
            x_vals = np.arange(x_min, x_max + 1e-9, sample_step)
            a, b = poly_coeff
            for xv in x_vals:
                yv = a * xv + b
                sampled_points.append((xv, yv))

            # 2) y좌표만 중력중심 보정
            refined = compute_center_of_gravity_y(gray_float, sampled_points, window_size=window_size)

            # 3) 보정된 점들로 다시 피팅 (x->y)
            rx = refined[:,0]
            ry = refined[:,1]
            if len(rx) < degree + 1:
                continue

            new_coefs = np.polyfit(rx, ry, degree)
            # domain min/max
            new_x_min, new_x_max = float(np.min(rx)), float(np.max(rx))
            abs_diff = abs(new_x_max - new_x_min)

            # eq 갱신
            rows_updated["equations"][label] = list(new_coefs) + [new_x_min, new_x_max, abs_diff]

            # 4) draw_points=True -> 샘플링해서 그리기
            if draw_points and out_img is not None:
                num_draw = max(100, len(rx))
                # 다시 x_min ~ x_max 100개정도
                x_samp = np.linspace(new_x_min, new_x_max, num_draw)
                y_samp = np.polyval(new_coefs, x_samp)

                for i in range(num_draw - 1):
                    pt1 = (int(round(x_samp[i])), int(round(y_samp[i])))
                    pt2 = (int(round(x_samp[i+1])), int(round(y_samp[i+1])))
                    cv2.line(out_img, pt1, pt2, (0, 255, 0), 1)  # 초록색

    # === (B) Cols 갱신 ===
    for label, eq in cols_updated.get("equations", {}).items():
        if len(eq) < degree + 3:
            continue

        if degree == 1:
            # eq = [a, b, yMin, yMax, dummy]
            poly_coeff = eq[:degree+1]  # [a, b]
            y_min = eq[degree+1]
            y_max = eq[degree+2]
            if y_max < y_min:
                continue

            # 1) (y_min ~ y_max) 0.5 픽셀 단위로 샘플링
            sampled_points = []
            y_vals = np.arange(y_min, y_max + 1e-9, sample_step)
            a, b = poly_coeff
            for yv in y_vals:
                xv = a * yv + b
                sampled_points.append((xv, yv))

            # 2) x좌표만 중력중심 보정
            refined = compute_center_of_gravity_x(gray_float, sampled_points, window_size=window_size)

            rx = refined[:,0]
            ry = refined[:,1]
            if len(ry) < degree + 1:
                continue

            new_coefs = np.polyfit(ry, rx, degree)
            # domain min/max
            new_y_min, new_y_max = float(np.min(ry)), float(np.max(ry))
            abs_diff = abs(new_y_max - new_y_min)

            # eq 갱신
            cols_updated["equations"][label] = list(new_coefs) + [new_y_min, new_y_max, abs_diff]

            if draw_points and out_img is not None:
                num_draw = max(100, len(ry))
                y_samp = np.linspace(new_y_min, new_y_max, num_draw)
                x_samp = np.polyval(new_coefs, y_samp)

                for i in range(num_draw - 1):
                    pt1 = (int(round(x_samp[i])), int(round(y_samp[i])))
                    pt2 = (int(round(x_samp[i+1])), int(round(y_samp[i+1])))
                    cv2.line(out_img, pt1, pt2, (255, 0, 0), 1)  # 파란색

    if draw_points:
        return out_img, rows_updated, cols_updated
    else:
        return None, rows_updated, cols_updated

def visualize_line_plot(input_img, rows, cols):
    """
    입력 이미지 위에 rows와 cols의 다항식 방정식대로 선을 그려 시각화하는 함수.

    Parameters:
        input_img (numpy.ndarray): 입력 이미지 (그레이스케일 또는 컬러).
        rows (dict): 행(row) 다항식 방정식이 저장된 딕셔너리.
                     형식: rows['equations'][label] = [a_n, a_(n-1), ..., a_0, x_min, x_max, abs_diff]
        cols (dict): 열(col) 다항식 방정식이 저장된 딕셔너리.
                     형식: cols['equations'][label] = [a_n, a_(n-1), ..., a_0, y_min, y_max, abs_diff]

    Returns:
        None. 단순히 이미지를 화면에 표시.
    """
    # 입력 이미지가 컬러인지 그레이스케일인지 확인
    if len(input_img.shape) == 2:
        gray_img = input_img.copy()
    else:
        gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

    # Matplotlib Figure 및 Axes 생성
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.imshow(gray_img, cmap='gray', origin='upper')
    ax1.set_title("Comparison of Original & Fitted Curves with Intersection Points")
    ax1.set_axis_off()  # 축 숨기기

    # === 행(Rows) 다항식 그리기 ===
    for idx, (row_label, equation) in enumerate(rows.get("equations", {}).items()):
        if len(equation) < 4:
            # 최소 4개 요소 필요: 계수 n+1개, x_min, x_max, abs_diff
            continue

        # 계수와 도메인 분리
        *poly_coeff, x_min, x_max, abs_diff = equation
        degree = len(poly_coeff) - 1  # 다항식 차수

        # 다항식 계수가 numpy.polyval과 호환되도록 그대로 사용
        # numpy.polyval은 높은 차수부터 계수를 입력받음
        poly_coeff_np = np.array(poly_coeff)

        # x 샘플링
        x_samp = np.linspace(x_min, x_max, 500)
        y_samp = np.polyval(poly_coeff_np, x_samp)

        # 다항식 그리기 (녹색 실선)
        if idx == 0:
            label = 'Rows'
        else:
            label = None  # 첫 번째 항목만 레이블 지정하여 범례 중복 방지
        ax1.plot(x_samp, y_samp, color='green', linewidth=1, label=label)

    # === 열(Cols) 다항식 그리기 ===
    for idx, (col_label, equation) in enumerate(cols.get("equations", {}).items()):
        if len(equation) < 4:
            # 최소 4개 요소 필요: 계수 n+1개, y_min, y_max, abs_diff
            continue

        # 계수와 도메인 분리
        *poly_coeff, y_min, y_max, abs_diff = equation
        degree = len(poly_coeff) - 1  # 다항식 차수

        # 다항식 계수가 numpy.polyval과 호환되도록 그대로 사용
        # numpy.polyval은 높은 차수부터 계수를 입력받음
        poly_coeff_np = np.array(poly_coeff)

        # y 샘플링
        y_samp = np.linspace(y_min, y_max, 500)
        x_samp = np.polyval(poly_coeff_np, y_samp)

        # 다항식 그리기 (파란색 실선)
        if idx == 0:
            label = 'Cols'
        else:
            label = None  # 첫 번째 항목만 레이블 지정하여 범례 중복 방지
        ax1.plot(x_samp, y_samp, color='blue', linewidth=1, label=label)

    # 범례 추가 (중복 방지)
    handles, labels_plot = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels_plot, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc='upper right')

    # 레이아웃 조정 및 표시
    plt.tight_layout(pad=0)
    plt.show()



def poly_intersection_solver(row_eq, col_eq, degree):
    """
    row_eq: list containing [a_n, a_(n-1), ..., a0, x_min, x_max, ...]
    col_eq: list containing [b_n, b_(n-1), ..., b0, y_min, y_max, ...]
    degree: degree of the polynomials

    Row polynomial: y = a_n*x^n + a_(n-1)*x^(n-1) + ... + a0
    Column polynomial: x = b_n*y^n + b_(n-1)*y^(n-1) + ... + b0

    Returns: (x_sol, y_sol) or None
    """
    row_coeff = row_eq[:degree+1]
    x_min, x_max = row_eq[degree+1], row_eq[degree+2]
    col_coeff = col_eq[:degree+1]
    y_min, y_max = col_eq[degree+1], col_eq[degree+2]

    def func(vars):
        x, y = vars[0], vars[1]
        f1 = y - np.polyval(row_coeff, x)
        f2 = x - np.polyval(col_coeff, y)
        return [f1, f2]

    x0_init = 0.5 * (x_min + x_max)
    y0_init = np.polyval(row_coeff, x0_init)

    sol = root(func, [x0_init, y0_init], method='hybr')
    if sol.success:
        x_sol, y_sol = sol.x[0], sol.x[1]
        if (x_min - 1e-3 <= x_sol <= x_max + 1e-3) and (y_min - 1e-3 <= y_sol <= y_max + 1e-3):
            return (x_sol, y_sol)
    return None

def find_and_assign_intersections_P(original_img, rows, cols, max_contour, draw_points=True, degree=3):
    """
    row의 다항식, col의 다항식을 이용해 교차점 (x, y)를 구한 뒤
    rows_updated, cols_updated['points']에 할당

    Parameters:
    degree (int): Degree of the polynomials
    """
    if max_contour is not None:
        rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(max_contour)
    else:
        rect_x, rect_y, rect_w, rect_h = 0, 0, original_img.shape[1], original_img.shape[0]

    img = original_img.copy()
    rows_updated = rows.copy()
    cols_updated = cols.copy()

    for row_name in rows_updated["points"]:
        rows_updated["points"][row_name] = []
    for col_name in cols_updated["points"]:
        cols_updated["points"][col_name] = []

    intersection_points = set()

    for row_name, row_eq in rows["equations"].items():
        if len(row_eq) < degree + 3:
            continue
        for col_name, col_eq in cols["equations"].items():
            if len(col_eq) < degree + 3:
                continue

            pt_sol = poly_intersection_solver(row_eq, col_eq, degree)
            if pt_sol is not None:
                x_sol, y_sol = pt_sol
                if (rect_x <= x_sol <= rect_x + rect_w) and (rect_y <= y_sol <= rect_y + rect_h):
                    x_i = float(x_sol)
                    y_i = float(y_sol)
                    rows_updated["points"][row_name].append((x_i, y_i))
                    cols_updated["points"][col_name].append((x_i, y_i))
                    intersection_points.add((x_i, y_i))

    if draw_points:
        for (px, py) in intersection_points:
            cv2.circle(img, (int(px), int(py)), 2, (255, 0, 255), -1)

    return img, rows_updated, cols_updated


def extract_rois(rows, cols):
    # Process rows
    rows_roi = []
    for equation in rows['equations']:
        coefficients = equation[:-2]
        x_min = equation[-2]
        x_max = equation[-1]
        
        # Define polynomial
        poly = np.poly1d(coefficients)
        
        # Derivative of the polynomial
        deriv = np.polyder(poly)
        
        # Find roots of the derivative (critical points)
        critical_points = deriv.roots
        # Filter real and within range critical points
        real_critical_points = [cp.real for cp in critical_points 
                                if np.isreal(cp) and x_min <= cp.real <= x_max]
        
        # Evaluate polynomial at critical points and endpoints
        x_eval = real_critical_points + [x_min, x_max]
        y_eval = poly(x_eval)
        
        # Determine y_min and y_max
        y_min = np.min(y_eval)
        y_max = np.max(y_eval)
        
        # Expand ROI by 3 pixels
        expanded_x_min = x_min - 3
        expanded_x_max = x_max + 3
        expanded_y_min = y_min - 3
        expanded_y_max = y_max + 3
        
        # Store the expanded ROI
        rows_roi.append([expanded_x_min, expanded_y_min, expanded_x_max, expanded_y_max])
    
    rows['roi'] = rows_roi
    
    # Process columns
    cols_roi = []
    for equation in cols['equations']:
        coefficients = equation[:-2]
        y_min = equation[-2]
        y_max = equation[-1]
        
        # Define polynomial
        poly = np.poly1d(coefficients)
        
        # Derivative of the polynomial
        deriv = np.polyder(poly)
        
        # Find roots of the derivative (critical points)
        critical_points = deriv.roots
        # Filter real and within range critical points
        real_critical_points = [cp.real for cp in critical_points 
                                if np.isreal(cp) and y_min <= cp.real <= y_max]
        
        # Evaluate polynomial at critical points and endpoints
        y_eval = real_critical_points + [y_min, y_max]
        x_eval = poly(y_eval)
        
        # Determine x_min and x_max
        x_min = np.min(x_eval)
        x_max = np.max(x_eval)
        
        # Expand ROI by 3 pixels
        expanded_x_min = x_min - 3
        expanded_x_max = x_max + 3
        expanded_y_min = y_min - 3
        expanded_y_max = y_max + 3
        
        # Store the expanded ROI
        cols_roi.append([expanded_x_min, expanded_y_min, expanded_x_max, expanded_y_max])
    
    cols['roi'] = cols_roi
    
    return rows, cols

def remove_arc7(rows, cols):
    """
    Removes labels from rows and cols where the fourth value in ['equations'][label] is greater than 7.

    Args:
        rows (dict): Dictionary with row labels as keys and equations as values.
        cols (dict): Dictionary with column labels as keys and equations as values.

    Returns:
        tuple: Updated (rows, cols) dictionaries after removing labels with fourth value > 7.
    """
    def remove_labels(data_dict, n):
        """
        Helper function to remove labels from a dictionary where the fourth value in ['equations'][label] > 7.
        """
        if 'equations' in data_dict:
            equations = data_dict['equations']
            labels_to_remove = [label for label, eq in equations.items() if len(eq) >= 4 and eq[3] > n]
            for label in labels_to_remove:
                print(f"  Removing label '{label}' from equations due to fourth value > {n}.")
                # Remove the label from equations
                del equations[label]
                # Remove corresponding points if they exist
                if label in data_dict.get('points', {}):
                    del data_dict['points'][label]
        return data_dict

    # Remove labels from rows
    rows = remove_labels(rows, n=10)
    
    # Remove labels from cols
    cols = remove_labels(cols, n=6)
    
    return rows, cols

def clean_and_relabel(rows, cols):
    # Helper function to process equations
    def process_equations(data, prefix):
        equations = data.get('equations', {})
        if not isinstance(equations, dict):
            print(f"Warning: 'equations' in {prefix} is not a dictionary.")
            return {}
        
        # Remove labels with [0, 0, 0, 0]
        filtered_equations = {label: value for label, value in equations.items() if value != [0, 0, 0, 0]}
        
        # Relabel sequentially
        relabeled = {}
        for i, (label, value) in enumerate(filtered_equations.items(), start=1):
            new_label = f"{prefix}{i}"
            relabeled[new_label] = value
        
        return relabeled
    
    # Helper function to process points
    def process_points(data, prefix):
        points = data.get('points', {})
        if not isinstance(points, dict):
            print(f"Warning: 'points' in {prefix} is not a dictionary.")
            return {}
        
        # Remove labels with empty values
        filtered_points = {label: value for label, value in points.items() if value}
        
        # Relabel sequentially
        relabeled = {}
        for i, (label, value) in enumerate(filtered_points.items(), start=1):
            new_label = f"{prefix}{i}"
            relabeled[new_label] = value
        
        return relabeled
    
    # Process rows
    # Process equations in rows
    rows['equations'] = process_equations(rows, 'row')
    # Process points in rows
    rows['points'] = process_points(rows, 'row')
    
    # Process columns
    # Process equations in columns
    cols['equations'] = process_equations(cols, 'col')
    # Process points in columns
    cols['points'] = process_points(cols, 'col')
    
    return rows, cols

def indexing_data(rows, cols, input_image, mask_contour, circle_radius, epsilon=1e-6):
    """
    rows, cols에 담긴 점들의 유효성을 먼저 검증한 후,
    각 점 주변의 10x10 영역 평균 밝기를 계산하여 중심점을 찾고,
    재인덱싱한 row, col 정보를 기반으로 grid intersection을 구성합니다.
    
    기존보다 각 태스크의 오버헤드를 줄이기 위해, 
    - 각 row/col의 유효 점을 미리 캐싱하고
    - executor.map을 사용하여 배치 처리합니다.
    """
    def validate_points(points, entity_label):
        valid_points = []
        invalid_point_count = 0
        for point in points:
            if (isinstance(point, (list, tuple)) and len(point) == 2 and 
                all(isinstance(coord, (int, float)) and not math.isnan(coord) and not math.isinf(coord) for coord in point)):
                valid_points.append(point)
            else:
                invalid_point_count += 1
        if invalid_point_count > 0:
            print(f"Skipped {invalid_point_count} invalid points in {entity_label}.")
        return valid_points

    def calculate_average_brightness(image, point):
        x, y = point
        half_size = int(circle_radius / 4.5)  # 10x10 영역 (중심 기준 ±10)
        x_start = max(0, int(x - half_size))
        x_end = min(image.shape[1], int(x + half_size))
        y_start = max(0, int(y - half_size))
        y_end = min(image.shape[0], int(y + half_size))
        roi = image[y_start:y_end, x_start:x_end]
        return np.mean(roi)

    def find_closest_entity(point, entities_points):
        min_distance = float('inf')
        closest_entity = None
        for label, points in entities_points.items():
            for entity_point in points:
                distance = math.hypot(point[0] - entity_point[0], point[1] - entity_point[1])
                if distance < min_distance:
                    min_distance = distance
                    closest_entity = label
        return closest_entity

    def find_closest_row(point, row_points_raw):
        min_distance = float('inf')
        closest_row_label = None
        for label, points in row_points_raw.items():
            for p in points:
                distance = math.hypot(point[0] - p[0], point[1] - p[1])
                if distance < min_distance:
                    min_distance = distance
                    closest_row_label = label
        return closest_row_label

    def find_closest_col(point, col_points_raw):
        min_distance = float('inf')
        closest_col_label = None
        for label, points in col_points_raw.items():
            for p in points:
                distance = math.hypot(point[0] - p[0], point[1] - p[1])
                if distance < min_distance:
                    min_distance = distance
                    closest_col_label = label
        return closest_col_label

    # 1. row_points의 유효 점들을 미리 검증 후 캐싱 및 각 row의 평균 y 계산
    row_points_raw = rows.get('points', {})
    validated_row_points = {}
    row_avg_y = {}
    for label, points in row_points_raw.items():
        valid_points = validate_points(points, f"row {label}")
        if valid_points:
            validated_row_points[label] = valid_points
            row_avg_y[label] = np.mean([p[1] for p in valid_points])
    if not row_avg_y:
        return json.dumps({'error': 'No valid row points to calculate brightness.'}), {}, {}, {}

    # 2. 가우시안 블러 적용 및 그레이스케일 변환 (밝기 계산을 위해)
    gaussian_image = cv2.GaussianBlur(input_image, (7, 7), 0)
    if len(gaussian_image.shape) > 2:
        gaussian_image = cv2.cvtColor(gaussian_image, cv2.COLOR_BGR2GRAY)

    # 3. 각 후보 점 주변의 평균 밝기 계산 (모든 valid point를 한 리스트로 모아서 batch 처리)
    brightness_items = []
    for label, points in validated_row_points.items():
        for point in points:
            brightness_items.append((label, point))
    
    def compute_brightness(item):
        label, point = item
        brightness = calculate_average_brightness(gaussian_image, point)
        return brightness, point, label

    with concurrent.futures.ThreadPoolExecutor() as executor:
        brightness_results = list(executor.map(compute_brightness, brightness_items))
    if not brightness_results:
        return json.dumps({'error': 'No points with brightness found.'}), {}, {}, {}

    # 센터 포인트: 10x10 영역 평균 밝기가 가장 큰 점 선택
    max_brightness, center_point, best_row_label = max(brightness_results, key=lambda t: t[0])

    # 4. center row/col 결정 (각각 가장 가까운 row, col label)
    center_row_label = find_closest_row(center_point, row_points_raw)
    center_col_label = find_closest_entity(center_point, cols.get('points', {}))
    if center_col_label is None:
        return json.dumps({'error': 'No columns found for the best point.'}), {}, {}, {}

    # center label에서 숫자만 추출하여 중심 인덱스로 활용 (예: "row26" -> 26)
    try:
        center_row_num = int(center_row_label.replace('row', ''))
    except Exception as e:
        center_row_num = 0
    try:
        center_col_num = int(center_col_label.replace('col', ''))
    except Exception as e:
        center_col_num = 0

    # 각 row, col label의 재인덱싱 (center를 0 기준으로)
    row_index_mapping = {}
    for label in row_points_raw.keys():
        try:
            row_index_mapping[label] = int(label.replace('row', '')) - center_row_num
        except Exception as e:
            row_index_mapping[label] = 0

    col_points_raw = cols.get('points', {})
    col_index_mapping = {}
    for label in col_points_raw.keys():
        try:
            col_index_mapping[label] = int(label.replace('col', '')) - center_col_num
        except Exception as e:
            col_index_mapping[label] = 0

    # 5. 그룹핑: 각 점이 속한 row와 col에 대해 재인덱스된 번호로 id 할당
    rows_dict = {}
    for old_label, points in validated_row_points.items():
        new_row_index = row_index_mapping.get(old_label, 0)
        new_row_key = f"row{new_row_index}"
        for point in points:
            x, y = point
            closest_col_label = find_closest_col(point, col_points_raw)
            new_col_index = col_index_mapping.get(closest_col_label, 0) if closest_col_label else 0
            point_id = (new_row_index, new_col_index)
            rows_dict.setdefault(new_row_key, []).append({
                "id": point_id,
                "x": x,
                "y": y
            })

    # col에 대해서도 미리 유효 점 캐싱
    validated_col_points = {}
    for label, points in col_points_raw.items():
        valid_points = validate_points(points, f"col {label}")
        if valid_points:
            validated_col_points[label] = valid_points

    cols_dict = {}
    for old_label, points in validated_col_points.items():
        new_col_index = col_index_mapping.get(old_label, 0)
        new_col_key = f"col{new_col_index}"
        for point in points:
            x, y = point
            closest_row_label = find_closest_row(point, row_points_raw)
            new_row_index = row_index_mapping.get(closest_row_label, 0) if closest_row_label else 0
            point_id = (new_row_index, new_col_index)
            cols_dict.setdefault(new_col_key, []).append({
                "id": point_id,
                "x": x,
                "y": y
            })

    # 6. Grid intersection 구성: 각 row와 col의 조합에 대해 가장 가까운 점 결정
    def process_row_intersections(old_row_label):
        valid_points = validated_row_points.get(old_row_label, [])
        local_points = []
        new_row_index = row_index_mapping.get(old_row_label, 0)
        for old_col_label, col_points in validated_col_points.items():
            try:
                avg_x_for_col = np.mean([p[0] for p in col_points])
            except Exception as e:
                avg_x_for_col = 0
            if valid_points:
                closest_point = min(valid_points, key=lambda p: abs(p[0] - avg_x_for_col))
                x, y = closest_point
                new_col_index = col_index_mapping.get(old_col_label, 0)
                local_points.append({
                    "id": (new_row_index, new_col_index),
                    "x": x,
                    "y": y
                })
        return local_points

    all_intersection_points = []
    # row 단위로 배치 처리
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_row_intersections, validated_row_points.keys()))
    for res in results:
        all_intersection_points.extend(res)
    points_list = all_intersection_points

    # 결과 조립 및 JSON 변환
    result_dict = {'points': points_list}
    result_json = {
        'center_point': list(center_point),
        'points': points_list
    }
    def np_converter(o):
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        else:
            raise TypeError("Object of type %s is not JSON serializable" % type(o))
    result_json_str = json.dumps(result_json, indent=4, default=np_converter)
    return result_json_str, result_dict, rows_dict, cols_dict, center_point


def find_min_slope_rows_json(rows, cols, input_image, epsilon=1e-6):
    def validate_points(points, entity_label):
        valid_points = []
        invalid_point_count = 0
        for point in points:
            if (isinstance(point, tuple) and len(point) == 2 and 
                all(isinstance(coord, (int, float)) and not math.isnan(coord) and not math.isinf(coord) for coord in point)):
                valid_points.append(point)
            else:
                invalid_point_count += 1
        if invalid_point_count > 0:
            print(f"Skipped {invalid_point_count} invalid points in {entity_label}.")
        return valid_points

    def calculate_average_brightness(image, point):
        x, y = point
        half_size = 5  # 10x10 영역 (중심 기준 ±5)
        x_start = max(0, int(x - half_size))
        x_end = min(image.shape[1], int(x + half_size))
        y_start = max(0, int(y - half_size))
        y_end = min(image.shape[0], int(y + half_size))
        roi = image[y_start:y_end, x_start:x_end]
        return np.mean(roi)

    def find_closest_entity(point, entities_points):
        min_distance = float('inf')
        closest_entity = None
        for label, points in entities_points.items():
            for entity_point in points:
                distance = math.hypot(point[0] - entity_point[0], point[1] - entity_point[1])
                if distance < min_distance:
                    min_distance = distance
                    closest_entity = label
        return closest_entity

    def find_closest_row(point, row_points_raw):
        min_distance = float('inf')
        closest_row_label = None
        for label, points in row_points_raw.items():
            for p in points:
                distance = math.hypot(point[0] - p[0], point[1] - p[1])
                if distance < min_distance:
                    min_distance = distance
                    closest_row_label = label
        return closest_row_label

    def find_closest_col(point, col_points_raw):
        min_distance = float('inf')
        closest_col_label = None
        for label, points in col_points_raw.items():
            for p in points:
                distance = math.hypot(point[0] - p[0], point[1] - p[1])
                if distance < min_distance:
                    min_distance = distance
                    closest_col_label = label
        return closest_col_label

    # 1. row_points 수집 및 유효성 검증, 그리고 각 row의 평균 y 계산
    row_points_raw = rows.get('points', {})
    row_avg_y = {}
    for label, points in row_points_raw.items():
        valid_points = validate_points(points, f"row {label}")
        if valid_points:
            avg_y = np.mean([p[1] for p in valid_points])
            row_avg_y[label] = avg_y

    if not row_avg_y:
        return json.dumps({'error': 'No valid row points to calculate brightness.'}), {}, {}, {}

    # 2. 가우시안 3x3 필터 적용 후 그레이스케일 변환 (밝기 계산을 위해)
    gaussian_image = cv2.GaussianBlur(input_image, (3, 3), 0)
    if len(gaussian_image.shape) > 2:
        gaussian_image = cv2.cvtColor(gaussian_image, cv2.COLOR_BGR2GRAY)

    # 3. 각 후보 점 주변 10x10 영역의 평균 밝기를 계산 (병렬 처리)
    brightness_results = []  # (평균 밝기, point, row_label)
    def compute_brightness(label, point):
        brightness = calculate_average_brightness(gaussian_image, point)
        return brightness, point, label

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for label, points in row_points_raw.items():
            valid_points = validate_points(points, f"row {label}")
            for point in valid_points:
                futures.append(executor.submit(compute_brightness, label, point))
        for future in concurrent.futures.as_completed(futures):
            brightness, point, label = future.result()
            brightness_results.append((brightness, point, label))
    if not brightness_results:
        return json.dumps({'error': 'No points with brightness found.'}), {}, {}, {}

    # 센터 포인트: 10x10 영역 평균 밝기가 가장 큰 점 선택
    max_brightness, center_point, best_row_label = max(brightness_results, key=lambda t: t[0])

    # 4. center row/col 결정 (center_row_label, center_col_label)
    def find_closest_row_for_center(point, row_points_raw):
        min_distance = float('inf')
        closest_row_label = None
        for label, points in row_points_raw.items():
            for p in points:
                distance = math.hypot(point[0] - p[0], point[1] - p[1])
                if distance < min_distance:
                    min_distance = distance
                    closest_row_label = label
        return closest_row_label

    center_row_label = find_closest_row_for_center(center_point, row_points_raw)
    center_col_label = find_closest_entity(center_point, cols.get('points', {}))
    if center_col_label is None:
        return json.dumps({'error': 'No columns found for the best point.'}), {}, {}, {}

    # ===== 기존 인덱싱 대신, label의 숫자 부분을 이용하여 재인덱싱 (center를 0으로) =====
    # center_row_label이 "row26", center_col_label이 "col26"라고 가정
    center_row_num = int(center_row_label.replace('row',''))
    center_col_num = int(center_col_label.replace('col',''))

    # 각 row, col label의 새 인덱스: 기존 숫자 - center 숫자
    row_index_mapping = {}
    for label in row_points_raw.keys():
        try:
            row_index_mapping[label] = int(label.replace('row','')) - center_row_num
        except Exception as e:
            print(f"Error converting row label {label}: {e}")

    col_points_raw = cols.get('points', {})  # 미리 추출
    col_index_mapping = {}
    for label in col_points_raw.keys():
        try:
            col_index_mapping[label] = int(label.replace('col','')) - center_col_num
        except Exception as e:
            print(f"Error converting col label {label}: {e}")

    # ===== 그룹핑: 각 점이 속한 row와 col에 대해 재인덱스된 번호를 이용하여 id: (row_index, col_index) 할당 =====

    # (a) rows_dict 구성: 각 row의 점들에 대해 새 row label을 사용
    rows_dict = {}
    for old_label, points in row_points_raw.items():
        valid_points = validate_points(points, f"row {old_label}")
        new_row_index = row_index_mapping.get(old_label, 0)
        new_row_key = f"row{new_row_index}"
        for point in valid_points:
            x, y = point
            # 점의 열은 가까운 col label을 이용하여 결정
            closest_col_label = find_closest_col(point, col_points_raw)
            new_col_index = col_index_mapping.get(closest_col_label, 0) if closest_col_label else 0
            point_id = (new_row_index, new_col_index)  # id: (i, j)
            if new_row_key not in rows_dict:
                rows_dict[new_row_key] = []
            rows_dict[new_row_key].append({
                "id": point_id,
                "x": x,
                "y": y
            })

    # (b) cols_dict 구성: 각 col의 점들에 대해 새 col label을 사용
    cols_dict = {}
    for old_label, points in col_points_raw.items():
        valid_points = validate_points(points, f"col {old_label}")
        new_col_index = col_index_mapping.get(old_label, 0)
        new_col_key = f"col{new_col_index}"
        for point in valid_points:
            x, y = point
            # 점의 행은 가까운 row label을 이용하여 결정
            closest_row_label = find_closest_row(point, row_points_raw)
            new_row_index = row_index_mapping.get(closest_row_label, 0) if closest_row_label else 0
            point_id = (new_row_index, new_col_index)
            if new_col_key not in cols_dict:
                cols_dict[new_col_key] = []
            cols_dict[new_col_key].append({
                "id": point_id,
                "x": x,
                "y": y
            })

    # (c) Grid intersection 구성: 각 row와 col의 조합에 대해 id를 (row_index, col_index)로 할당
    points_list = []
    def process_row_intersections(old_row_label):
        valid_row_points = validate_points(row_points_raw.get(old_row_label, []), f"row {old_row_label}")
        local_points = []
        new_row_index = row_index_mapping.get(old_row_label, 0)
        for old_col_label in col_points_raw.keys():
            # 각 col의 평균 x 값은 여기서 간단히 계산하거나, 이미 계산된 값을 사용할 수 있음
            try:
                col_points = validate_points(col_points_raw.get(old_col_label, []), f"col {old_col_label}")
                avg_x_for_col = np.mean([p[0] for p in col_points]) if col_points else 0
            except Exception as e:
                avg_x_for_col = 0
            if valid_row_points:
                # row 내에서 해당 col의 평균 x와 가장 가까운 점 선택
                closest_point = min(valid_row_points, key=lambda p: abs(p[0]-avg_x_for_col))
                x, y = closest_point
                new_col_index = col_index_mapping.get(old_col_label, 0)
                local_points.append({
                    "id": (new_row_index, new_col_index),  # id: (i, j)
                    "x": x,
                    "y": y
                })
        return local_points

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_row_intersections, old_row_label) for old_row_label in row_points_raw.keys()]
        for future in concurrent.futures.as_completed(futures):
            points_list.extend(future.result())

    # 결과 조립
    result_dict = {
        'points': points_list
    }
    result_json = {
        'center_point': list(center_point),
        'points': points_list
    }
    def np_converter(o):
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        else:
            raise TypeError("Object of type %s is not JSON serializable" % type(o))
    
    result_json_str = json.dumps(result_json, indent=4, default=np_converter)
    return result_json_str, result_dict, rows_dict, cols_dict, center_point





def get_col_avg_x(cols):
    """
    Computes the average x-coordinate for each column label.

    Args:
        cols (dict): Columns dictionary.

    Returns:
        dict: Mapping of column labels to their average x-coordinate.
    """
    col_avg = {}
    for label, points in cols.items():
        if points:
            avg_x = sum(p['x'] for p in points) / len(points)
            col_avg[label] = avg_x
        else:
            col_avg[label] = 0  # Default value if no points
    return col_avg

def get_row_avg_y(rows):
    """
    Computes the average y-coordinate for each row label.

    Args:
        rows (dict): Rows dictionary.

    Returns:
        dict: Mapping of row labels to their average y-coordinate.
    """
    row_avg = {}
    for label, points in rows.items():
        if points:
            avg_y = sum(p['y'] for p in points) / len(points)
            row_avg[label] = avg_y
        else:
            row_avg[label] = 0  # Default value if no points
    return row_avg

def calculate_slope_angles(labels_dict, is_column=True):
    """
    Calculates slope angles between consecutive labels based on their points.

    Args:
        labels_dict (dict): Dictionary containing labels with lists of points.
        is_column (bool): True if processing columns, False for rows.

    Returns:
        tuple: (list of slope angles, sorted list of labels)
    """
    slope_angles = []
    sorted_labels = []
    
    # Sorting labels based on average coordinate
    if is_column:
        sorted_labels = sorted(labels_dict.keys(), key=lambda l: get_col_avg_x({l: labels_dict[l]})[l])
    else:
        sorted_labels = sorted(labels_dict.keys(), key=lambda l: get_row_avg_y({l: labels_dict[l]})[l])

    # Calculate slope angle for each label based on first and last points
    label_slopes = {}
    for label in sorted_labels:
        points = labels_dict[label]
        if len(points) >= 2:
            sorted_points = sorted(points, key=lambda p: p['x'] if is_column else p['y'])
            x1, y1 = sorted_points[0]['x'], sorted_points[0]['y']
            x2, y2 = sorted_points[-1]['x'], sorted_points[-1]['y']
            if x2 - x1 != 0:
                slope = (y2 - y1) / (x2 - x1)
                angle = math.degrees(math.atan(slope))
            else:
                angle = 90.0  # Vertical line
            label_slopes[label] = angle
            slope_angles.append(angle)
        elif len(points) == 1:
            # If only one point, define slope as 0 degrees
            label_slopes[label] = 0.0
            slope_angles.append(0.0)
        else:
            # If no points, define slope as 0 degrees
            label_slopes[label] = 0.0
            slope_angles.append(0.0)
    
    return slope_angles, sorted_labels, label_slopes

def remove_first_last_labels(rows_removed, cols_removed, row_n, col_n, result_removed=None):
    """
    행과 열의 첫 번째와 마지막 n개의 레이블을 제거하고, 관련된 포인트들도 함께 제거합니다.

    Parameters:
    - rows_removed: 행 레이블을 키로 하고, 포인트 리스트를 값으로 가지는 딕셔너리
    - cols_removed: 열 레이블을 키로 하고, 포인트 리스트를 값으로 가지는 딕셔너리
    - n: 제거할 레이블의 수
    - result_removed: 제거된 포인트를 저장할 딕셔너리 (선택사항)
    """
    # 정규 표현식 패턴 정의
    pattern = r'\((\-?\d+),(\-?\d+)\)'

    # 열 처리
    col_avg_x = get_col_avg_x(cols_removed)
    sorted_cols_final = sorted(col_avg_x.keys(), key=lambda l: col_avg_x[l])

    
    if len(sorted_cols_final) >= 2 * col_n:
        cols_to_remove = sorted_cols_final[:col_n] + sorted_cols_final[-col_n:]
    else:
        cols_to_remove = sorted_cols_final.copy()

    for col_label in cols_to_remove:
        if col_label in cols_removed:
            points_to_remove = cols_removed.pop(col_label)
            # print(f"  열 '{col_label}'을(를) 제거합니다.")

            ids_to_remove = [p['id'] for p in points_to_remove]

            # 관련된 행에서 포인트 제거
            for row_label, row_points in rows_removed.items():
                filtered_points = [p for p in row_points if p['id'] not in ids_to_remove]
                rows_removed[row_label] = filtered_points

            # 제거된 포인트를 result_removed에 저장
            if result_removed is not None:
                if 'columns' not in result_removed:
                    result_removed['columns'] = {}
                result_removed['columns'][col_label] = points_to_remove

    # 행 처리
    row_avg_y = get_row_avg_y(rows_removed)
    sorted_rows_final = sorted(row_avg_y.keys(), key=lambda l: row_avg_y[l])

    if len(sorted_rows_final) >= 2 * row_n:
        rows_to_remove = sorted_rows_final[:row_n] + sorted_rows_final[-row_n:]
    else:
        rows_to_remove = sorted_rows_final.copy()

    for row_label in rows_to_remove:
        if row_label in rows_removed:
            points_to_remove = rows_removed.pop(row_label)
            # print(f"  행 '{row_label}'을(를) 제거합니다.")

            ids_to_remove = [p['id'] for p in points_to_remove]

            # 관련된 열에서 포인트 제거
            for col_label, col_points in cols_removed.items():
                filtered_points = [p for p in col_points if p['id'] not in ids_to_remove]
                cols_removed[col_label] = filtered_points

            # 제거된 포인트를 result_removed에 저장
            if result_removed is not None:
                if 'rows' not in result_removed:
                    result_removed['rows'] = {}
                result_removed['rows'][row_label] = points_to_remove

    if result_removed is not None:
        return result_removed


def interval_based_anomaly_removal_columns(cols_removed, result_removed, rows_removed, epsilon, excluded_labels):
    """
    Performs interval-based anomaly removal on columns based on average x values.
    Removes a column if its actual average x is smaller than (1 - epsilon) * predicted average x.

    Args:
        cols_removed (dict): Columns dictionary to modify.
        result_removed (dict): Result dictionary to modify.
        rows_removed (dict): Rows dictionary to modify.
        epsilon (float): Threshold ratio for average x reduction to mark exclusion.
        excluded_labels (set): Set to track excluded labels.
    """
    iteration = 0
    max_iterations = 1000  # Prevent infinite loops

    while iteration < max_iterations:
        iteration += 1
        print(f"\nInterval-Based Anomaly Removal for Columns - Iteration {iteration}:")
        
        # Get current columns sorted by average x
        col_avg_x = get_col_avg_x(cols_removed)
        sorted_cols = sorted(col_avg_x.keys(), key=lambda l: col_avg_x[l])

        # Extract sorted average x values
        sorted_avg_x = [col_avg_x[l] for l in sorted_cols]

        # Flag to check if any exclusion happens in this iteration
        exclusion_occurred = False

        # Iterate over sorted columns starting from the third one for prediction
        for i in range(5, len(sorted_cols)):
            pred = 2 * sorted_avg_x[i-1] - sorted_avg_x[i-2]  # Linear prediction based on previous two averages
            actual = sorted_avg_x[i]

            # Prepare labels and average x for i-1, i, i+1
            label_i_minus_1 = sorted_cols[i-1]
            label_i = sorted_cols[i]
            if i+1 < len(sorted_cols):
                label_i_plus_1 = sorted_cols[i+1]
                avg_x_i_plus_1 = f"{sorted_avg_x[i+1]:.4f}"
            else:
                label_i_plus_1 = 'N/A'
                avg_x_i_plus_1 = 'N/A'

            print(f"  Column {i}: Predicted Avg X = {pred:.4f}, Actual Avg X = {actual:.4f}")
            print(f"    Labels: {label_i_minus_1} (Avg X = {sorted_avg_x[i-1]:.4f}), "
                  f"{label_i} (Avg X = {sorted_avg_x[i]:.4f}), "
                  f"{label_i_plus_1} (Avg X = {avg_x_i_plus_1})")

            # Check if the actual average x is smaller than (1 - epsilon) * predicted average x
            if (pred - actual) > 5:
                if i < len(sorted_cols):  # sorted_cols[i] is the column to exclude
                    exclude_label = sorted_cols[i]
                    if exclude_label in excluded_labels:
                        print(f"  Column '{exclude_label}' is already excluded. Skipping.")
                        continue
                    print(f"  Detected anomaly. Marking column '{exclude_label}' for exclusion.")

                    # Remove the column from cols_removed
                    points_to_remove = cols_removed.pop(exclude_label, None)

                    if points_to_remove:
                        excluded_labels.add(exclude_label)
                        num_points = len(points_to_remove)
                        print(f"  Excluding column '{exclude_label}' and removing {num_points} points.")

                        # Collect all point IDs to remove
                        ids_to_remove = [p['id'] for p in points_to_remove]

                        # Remove points from rows_removed
                        for row_label, row_points in rows_removed.items():
                            filtered_points = [p for p in row_points if p['id'] not in ids_to_remove]
                            rows_removed[row_label] = filtered_points

                        # Remove points from result_removed
                        result_removed['points'] = [p for p in result_removed['points'] if p['id'] not in ids_to_remove]

                        exclusion_occurred = True
                        break  # Remove only the first detected anomaly per iteration
                    else:
                        print(f"  Warning: Column '{exclude_label}' not found in cols_removed.")
                else:
                    print(f"  Warning: Interval index {i} exceeds sorted_cols length.")

        if not exclusion_occurred:
            print("  No more anomalies detected for columns. Stopping interval-based removal.")
            break  # Exit the loop if no exclusion occurred

    else:
        print("  Reached maximum iterations for columns. Stopping interval-based removal.")


def interval_based_anomaly_removal_rows(rows_removed, result_removed, cols_removed, epsilon, excluded_labels):
    """
    Performs interval-based anomaly removal on rows based on average y values.
    Removes a row if its actual average y is smaller than (1 - epsilon) * predicted average y.

    Args:
        rows_removed (dict): Rows dictionary to modify.
        result_removed (dict): Result dictionary to modify.
        cols_removed (dict): Columns dictionary to modify.
        epsilon (float): Threshold ratio for average y reduction to mark exclusion.
        excluded_labels (set): Set to track excluded labels.
    """
    iteration = 0
    max_iterations = 1000  # Prevent infinite loops

    while iteration < max_iterations:
        iteration += 1
        print(f"\nInterval-Based Anomaly Removal for Rows - Iteration {iteration}:")
        
        # Get current rows sorted by average y
        row_avg_y = get_row_avg_y(rows_removed)
        sorted_rows = sorted(row_avg_y.keys(), key=lambda l: row_avg_y[l])

        # Extract sorted average y values
        sorted_avg_y = [row_avg_y[l] for l in sorted_rows]

        # Flag to check if any exclusion happens in this iteration
        exclusion_occurred = False

        # Iterate over sorted rows starting from the third one for prediction
        for i in range(2, len(sorted_rows)):
            pred = 2 * sorted_avg_y[i-1] - sorted_avg_y[i-2]  # Linear prediction based on previous two averages
            actual = sorted_avg_y[i]

            # Prepare labels and average y for i-1, i, i+1
            label_i_minus_1 = sorted_rows[i-1]
            label_i = sorted_rows[i]
            if i+1 < len(sorted_rows):
                label_i_plus_1 = sorted_rows[i+1]
                avg_y_i_plus_1 = f"{sorted_avg_y[i+1]:.4f}"
            else:
                label_i_plus_1 = 'N/A'
                avg_y_i_plus_1 = 'N/A'

            print(f"  Row {i}: Predicted Avg Y = {pred:.4f}, Actual Avg Y = {actual:.4f}")
            print(f"    Labels: {label_i_minus_1} (Avg Y = {sorted_avg_y[i-1]:.4f}), "
                  f"{label_i} (Avg Y = {sorted_avg_y[i]:.4f}), "
                  f"{label_i_plus_1} (Avg Y = {avg_y_i_plus_1})")

            # Check if the actual average y is smaller than (1 - epsilon) * predicted average y
            if actual < (1 - epsilon) * pred:
                if i < len(sorted_rows):  # sorted_rows[i] is the row to exclude
                    exclude_label = sorted_rows[i]
                    if exclude_label in excluded_labels:
                        print(f"  Row '{exclude_label}' is already excluded. Skipping.")
                        continue
                    print(f"  Detected anomaly. Marking row '{exclude_label}' for exclusion.")

                    # Remove the row from rows_removed
                    points_to_remove = rows_removed.pop(exclude_label, None)

                    if points_to_remove:
                        excluded_labels.add(exclude_label)
                        num_points = len(points_to_remove)
                        print(f"  Excluding row '{exclude_label}' and removing {num_points} points.")

                        # Collect all point IDs to remove
                        ids_to_remove = [p['id'] for p in points_to_remove]

                        # Remove points from cols_removed
                        for col_label, col_points in cols_removed.items():
                            filtered_points = [p for p in col_points if p['id'] not in ids_to_remove]
                            cols_removed[col_label] = filtered_points

                        # Remove points from result_removed
                        result_removed['points'] = [p for p in result_removed['points'] if p['id'] not in ids_to_remove]

                        exclusion_occurred = True
                        break  # Remove only the first detected anomaly per iteration
                    else:
                        print(f"  Warning: Row '{exclude_label}' not found in rows_removed.")
                else:
                    print(f"  Warning: Interval index {i} exceeds sorted_rows length.")

        if not exclusion_occurred:
            print("  No more anomalies detected for rows. Stopping interval-based removal.")
            break  # Exit the loop if no exclusion occurred

    else:
        print("  Reached maximum iterations for rows. Stopping interval-based removal.")



def slope_based_anomaly_removal(
    labels_dict,
    removed_dict,
    result_removed,
    rows_removed,
    cols_removed,
    slope_threshold,
    is_column=True,
    excluded_labels=None
):
    """
    Removes labels based on slope angle prediction deviations and prints slopes of i-1 and i labels.
    Performs the anomaly detection in both forward and reverse directions.

    Args:
        labels_dict (dict): Dictionary containing labels with lists of points.
        removed_dict (dict): Dictionary to remove points from based on label removal.
        result_removed (dict): Result dictionary to modify.
        rows_removed (dict): Rows dictionary to modify.
        cols_removed (dict): Columns dictionary to modify.
        slope_threshold (float): Threshold for slope angle deviation to mark exclusion.
        is_column (bool): True if processing columns, False for rows.
        excluded_labels (set): Set to track excluded labels.
    """
    if excluded_labels is None:
        excluded_labels = set()

    def process_direction(slope_angles, sorted_labels, direction='forward'):
        """
        Processes the slope angles in the specified direction to detect and remove anomalies.

        Args:
            slope_angles (list): List of slope angles.
            sorted_labels (list): List of sorted labels corresponding to slope angles.
            direction (str): 'forward' or 'reverse' indicating processing direction.
        """
        def smallest_angle_diff(a, b):
            diff = abs(a - b)
            return min(diff, 360 - diff)

        n = len(slope_angles)
        if n < 2:
            return  # Not enough data to perform anomaly detection

        if direction == 'forward':
            indices = range(1, n)
        elif direction == 'reverse':
            indices = range(n - 2, -1, -1)
        else:
            raise ValueError("Invalid direction. Use 'forward' or 'reverse'.")

        # Initialize previous difference
        if direction == 'forward':
            pred_slope_prev = slope_angles[0]
            actual_slope_prev = slope_angles[1]
            diff_prev = smallest_angle_diff(pred_slope_prev, actual_slope_prev)
        else:
            pred_slope_prev = slope_angles[-1]
            actual_slope_prev = slope_angles[-2]
            diff_prev = smallest_angle_diff(pred_slope_prev, actual_slope_prev)

        for i in indices:
            if direction == 'forward':
                if i >= 2:
                    pred_slope = 2 * slope_angles[i - 1] - slope_angles[i - 2]
                else:
                    pred_slope = slope_angles[i - 1]

                actual_slope = slope_angles[i]
            else:  # reverse
                if i < n - 2:
                    pred_slope = 2 * slope_angles[i + 1] - slope_angles[i + 2]
                else:
                    pred_slope = slope_angles[i + 1]

                actual_slope = slope_angles[i]

            diff_current = smallest_angle_diff(pred_slope, actual_slope)

            # Print the required information
            print(f"  {'Column' if is_column else 'Row'} Slope Angle Difference {i}: "
                  f"Prev Slope = {slope_angles[i - 1]:.2f}°, " if direction == 'forward' else
                  f"Next Slope = {slope_angles[i + 1]:.2f}°, ",
                  end='')
            print(f"Current Slope = {slope_angles[i]:.2f}°, "
                  f"Predicted Slope = {pred_slope:.2f}°, "
                  f"Prev Diff = {diff_prev:.2f}°, "
                  f"Current Diff = {diff_current:.2f}°")

            # Apply the anomaly detection condition
            if (diff_current < 0.5 * diff_prev or diff_current > 1.5 * diff_prev) and diff_current > slope_threshold:
                # Determine the label to exclude
                exclude_index = i
                if direction == 'reverse':
                    exclude_index = i
                if 0 <= exclude_index < len(sorted_labels):
                    exclude_label = sorted_labels[exclude_index]
                    if exclude_label not in excluded_labels:
                        # Retrieve slopes of relevant labels
                        if direction == 'forward':
                            slope_prev = slope_angles[i - 1]
                            slope_current = slope_angles[i]
                            slope_next = slope_angles[i + 1] if (i + 1) < len(slope_angles) else None
                        else:
                            slope_prev = slope_angles[i + 1]
                            slope_current = slope_angles[i]
                            slope_next = slope_angles[i - 1] if (i - 1) >= 0 else None

                        print(f"  Detected {'column' if is_column else 'row'} slope angle difference anomaly between "
                              f"'{sorted_labels[i - 1]}' (Slope: {slope_prev:.2f}°) and "
                              f"'{sorted_labels[i]}' (Slope: {slope_current:.2f}°).")
                        if slope_next is not None:
                            print(f"  {'Next' if direction == 'forward' else 'Previous'} label "
                                  f"'{sorted_labels[i + 1] if direction == 'forward' else sorted_labels[i - 1]}' "
                                  f"has Slope: {slope_next:.2f}°.")
                        print(f"  Marking '{exclude_label}' for exclusion.")
                        points_to_remove = removed_dict.pop(exclude_label, None)
                        if points_to_remove:
                            excluded_labels.add(exclude_label)
                            num_points = len(points_to_remove)
                            print(f"  Excluding {'column' if is_column else 'row'} '{exclude_label}' and removing {num_points} points.")

                            # Collect all point IDs to remove
                            ids_to_remove = [p['id'] for p in points_to_remove]

                            # Remove points from the opposite dictionary
                            if is_column:
                                # Remove from rows_removed
                                for row_label, row_points in rows_removed.items():
                                    filtered_points = [p for p in row_points if p['id'] not in ids_to_remove]
                                    rows_removed[row_label] = filtered_points
                            else:
                                # Remove from cols_removed
                                for col_label, col_points in cols_removed.items():
                                    filtered_points = [p for p in col_points if p['id'] not in ids_to_remove]
                                    cols_removed[col_label] = filtered_points

                            # Remove points from result_removed
                            result_removed['points'] = [p for p in result_removed['points'] if p['id'] not in ids_to_remove]

                            break  # Remove only the first detected anomaly per iteration
                else:
                    print(f"  Warning: Cannot determine {'column' if is_column else 'row'} to exclude for index {i}.")

            # Update diff_prev for the next iteration
            diff_prev = diff_current

    # 계산된 슬로프 각도와 정렬된 라벨 가져오기
    slope_angles, sorted_labels, label_slopes = calculate_slope_angles(labels_dict, is_column=is_column)

    # 슬로프 각도의 절대값으로 변환
    slope_angles = [abs(angle) for angle in slope_angles]

    # 정방향 처리
    print("Starting forward anomaly detection...")
    process_direction(slope_angles, sorted_labels, direction='forward')

    # 역방향 처리
    print("\nStarting reverse anomaly detection...")
    process_direction(slope_angles, sorted_labels, direction='reverse')


def slope_based_anomaly_removal_all(rows_removed, cols_removed, result_removed, slope_threshold, excluded_labels):
    """
    Performs slope-based anomaly removal for both rows and columns.

    Args:
        rows_removed (dict): Rows dictionary to modify.
        cols_removed (dict): Columns dictionary to modify.
        result_removed (dict): Result dictionary to modify.
        slope_threshold (float): Threshold for slope angle deviation to mark exclusion.
        excluded_labels (set): Set to track excluded labels.
    """
    print("\nSlope-Based Anomaly Removal for Columns:")
    slope_based_anomaly_removal(
        labels_dict=cols_removed,
        removed_dict=cols_removed,
        result_removed=result_removed,
        rows_removed=rows_removed,
        cols_removed=cols_removed,
        slope_threshold=slope_threshold,
        is_column=True,
        excluded_labels=excluded_labels
    )

    print("\nSlope-Based Anomaly Removal for Rows:")
    slope_based_anomaly_removal(
        labels_dict=rows_removed,
        removed_dict=rows_removed,
        result_removed=result_removed,
        rows_removed=rows_removed,
        cols_removed=cols_removed,
        slope_threshold=slope_threshold,
        is_column=False,
        excluded_labels=excluded_labels
    )

def remove_line(result_dict, rows_dict, cols_dict, epsilon=0.4, n=1, slope_threshold=15):
    """
    Iteratively removes column labels from cols_dict based on interval prediction deviations.
    After removal, it also removes the first n rows, last n rows, first n columns, and last n columns.
    Additionally, removes anomalies based on slope angle predictions for rows and columns.

    Args:
        result_dict (dict): Dictionary containing 'points' as a list of dicts with 'id', 'x', 'y'.
        rows_dict (dict): Dictionary with row labels as keys and lists of points as values.
        cols_dict (dict): Dictionary with column labels as keys and lists of points as values.
        epsilon (float, optional): Threshold ratio for interval deviation to mark exclusion.
                                   Defaults to 0.5 (i.e., 50%).
        n (int, optional): Number of rows and columns to remove from the start and end.
                           Defaults to 1.
        slope_threshold (float, optional): Threshold for slope angle deviation to mark exclusion.
                                          Defaults to 15 degrees.

    Returns:
        tuple: Updated (result_removed, rows_removed, cols_removed) dictionaries after removals.
    """
    # Deep copy to avoid modifying the original dictionaries
    result_removed = copy.deepcopy(result_dict)
    rows_removed = copy.deepcopy(rows_dict)
    cols_removed = copy.deepcopy(cols_dict)

    # Initialize excluded_labels as a set to track removed labels
    excluded_labels = set()

    # # === Step 1: Slope-Based Anomaly Removal ===
    # slope_based_anomaly_removal_all(rows_removed, cols_removed, result_removed, slope_threshold, excluded_labels)

    # # === Step 2: Interval-Based Anomaly Removal ===
    # interval_based_anomaly_removal_columns(cols_removed, result_removed, rows_removed, epsilon, excluded_labels)
    # interval_based_anomaly_removal_rows(rows_removed, result_removed, cols_removed, epsilon, excluded_labels)


    

    # === Step 3: Remove first and last n rows and columns ===
    # print(f"\nRemoving the first and last {n} rows and columns.")

    # Remove first and last n columns
    col_avg_x = get_col_avg_x(cols_removed)
    sorted_cols_final = sorted(col_avg_x.keys(), key=lambda l: col_avg_x[l])
   

    # Remove first and last n rows
    row_avg_y = get_row_avg_y(rows_removed)
    sorted_rows_final = sorted(row_avg_y.keys(), key=lambda l: row_avg_y[l])
    remove_first_last_labels(rows_removed, cols_removed, col_n=1, row_n=1, result_removed=result_removed)


    # print("\nFinal removal process completed.")

    return result_removed, rows_removed, cols_removed


def draw_points(original_img, rows_removed, cols_removed, center_point):
    """
    Draws points from rows_removed and cols_removed on separate images with unique colors per label.

    Args:
        original_img (numpy.ndarray): The original image in BGR format.
        rows_removed (dict): Dictionary with row labels as keys and lists of points as values.
        cols_removed (dict): Dictionary with column labels as keys and lists of points as values.

    Returns:
        tuple: (row_img, col_img) with points drawn.
    """
    # Create copies of the original image for rows and columns
    row_img = original_img.copy()
    col_img = original_img.copy()

    # Generate distinct colors for rows and columns
    # Using HSV color space for better color distribution
    def generate_colors(num_colors):
        colors = []
        for i in range(num_colors):
            hue = int(180 * i / num_colors)  # OpenCV uses hue range [0,179]
            saturation = 200 + random.randint(0, 55)  # Saturation between 200-255
            value = 200 + random.randint(0, 55)       # Value between 200-255
            color = cv2.cvtColor(np.uint8([[[hue, saturation, value]]]), cv2.COLOR_HSV2BGR)[0][0].tolist()
            colors.append(tuple(color))
        return colors

    # Assign unique colors to each row label
    row_labels = sorted(rows_removed.keys())
    num_rows = len(row_labels)
    row_colors = generate_colors(num_rows)
    row_color_dict = {label: color for label, color in zip(row_labels, row_colors)}

    # Assign unique colors to each column label
    col_labels = sorted(cols_removed.keys())
    num_cols = len(col_labels)
    col_colors = generate_colors(num_cols)
    col_color_dict = {label: color for label, color in zip(col_labels, col_colors)}

    # Define drawing parameters
    point_radius = 3
    point_thickness = -1  # Filled circle

    # Draw rows_removed points on row_img
    for label, points in rows_removed.items():
        color = row_color_dict.get(label, (0, 255, 0))  # Default to green if not found
        for point in points:
            x = int(point['x'])
            y = int(point['y'])
            cv2.circle(row_img, (x, y), point_radius, color, point_thickness)

    # Draw cols_removed points on col_img
    for label, points in cols_removed.items():
        color = col_color_dict.get(label, (255, 0, 0))  # Default to blue if not found
        for point in points:
            x = int(point['x'])
            y = int(point['y'])
            cv2.circle(col_img, (x, y), point_radius, color, point_thickness)

    cv2.circle(col_img, (int(center_point[0]), int(center_point[1])), 5, (255,0,0), 1)

    return row_img, col_img

def make_json(center_point, rows_removed):
    """
    center_point: 튜플 또는 리스트 형태의 중심 좌표, 예: (0, 0)
    rows_removed: 딕셔너리 형태, 각 키는 리스트 형태의 포인트들을 포함
                  각 포인트는 'id', 'x', 'y' 키를 가진 딕셔너리
    """
    final_json = {}
    final_json["center_point"] = list(center_point)
    
    # rows_removed가 딕셔너리인지 확인
    if not isinstance(rows_removed, dict):
        raise TypeError("rows_removed는 딕셔너리여야 합니다.")
    
    # 모든 포인트를 하나의 리스트로 수집
    points = []
    pattern = r'\((\-?\d+),\s*(\-?\d+)\)'  # 정규 표현식 패턴 정의
    
    for label, point_list in rows_removed.items():
        if not isinstance(point_list, list):
            raise TypeError(f"rows_removed['{label}']는 리스트여야 합니다.")
        for point in point_list:
            if not isinstance(point, dict):
                raise TypeError(f"rows_removed['{label}']의 각 포인트는 딕셔너리여야 합니다. 현재 타입: {type(point)}")
            if not all(k in point for k in ('id', 'x', 'y')):
                raise KeyError(f"포인트가 'id', 'x', 'y' 키를 모두 가지고 있는지 확인하십시오. 문제 포인트: {point}")
            
            # 'id', 'x', 'y' 값을 그대로 유지
            points.append(point)
    
    if not points:
        raise ValueError("rows_removed에 유효한 포인트가 없습니다.")
    
    # 포인트를 정렬하기 위해 id_x와 id_y 추출
    sorted_points = []
    for point in points:
        id_str = str(point['id'])
        match = re.match(pattern, id_str)
        if not match:
            raise ValueError(f"포인트의 'id' 형식이 잘못되었습니다. 예상 형식 '(id_x,id_y)', 실제: '{id_str}'")
        id_x = int(match.group(1))
        id_y = int(match.group(2))
        sorted_points.append((id_x, id_y, point))
    
    # id_x 오름차순, 그 다음 id_y 오름차순으로 정렬
    sorted_points.sort(key=lambda x: (x[0], x[1]))
    
    # 정렬된 포인트 리스트 추출
    sorted_point_dicts = [p[2] for p in sorted_points]
    
    # final_json에 points 할당
    final_json["points"] = sorted_point_dicts
    
    # JSON 문자열로 반환
    return json.dumps(final_json, indent=4, ensure_ascii=False)

def generate_distinct_colors(n_colors):
    """Generate n_distinct colors in RGB format."""
    colors = plt.cm.get_cmap('hsv', n_colors)  # 'hsv' 색상 공간에서 n개의 색상 생성
    return [tuple(int(c * 255) for c in colors(i)[:3]) for i in range(n_colors)]

def detect_ridges(gray, sigma=1.0):
    # Hessian matrix와 그 고유값을 이용하여 ridge(극값) 검출
    H_elems = hessian_matrix(gray, sigma=sigma, order='rc')
    maxima_ridges, minima_ridges = hessian_matrix_eigvals(H_elems)
    return maxima_ridges, minima_ridges

def sauvola_threshold_fast(image, window_size=15, k=0.5, R=128):
    """
    빠른 Sauvola thresholding 함수.
    cv2.boxFilter (C++ 가속) 를 사용하여, 지정된 window_size 내에서 
    로컬 평균과 분산을 계산한 후 Sauvola 공식에 따라 threshold를 계산합니다.
    
    image : 입력 이미지 (np.float64 형식 권장)
    window_size : 로컬 윈도우 크기 (예: 15)
    k : Sauvola 파라미터 (일반적으로 0.5 정도, 필요에 따라 조정)
    R : 표준편차의 동적 범위 (보통 128 또는 256, 이미지에 따라 조정)
    """
    # 입력 이미지를 float64로 변환 (필요한 경우)
    image = image.astype(np.float64)
    
    # cv2.boxFilter를 사용하여 로컬 평균 계산 (C++에서 최적화되어 빠름)
    mean = cv2.boxFilter(image, ddepth=-1, ksize=(window_size, window_size), borderType=cv2.BORDER_REPLICATE)
    # 제곱한 값의 로컬 평균 계산
    mean_sq = cv2.boxFilter(image * image, ddepth=-1, ksize=(window_size, window_size), borderType=cv2.BORDER_REPLICATE)
    
    # 분산과 표준편차 계산
    variance = mean_sq - mean * mean
    variance[variance < 0] = 0  # 오차로 음수값이 나오면 0으로 보정
    std = np.sqrt(variance)
    
    # Sauvola 공식: T = m * (1 + k*((std / R) - 1))
    threshold = mean * (1 + k * ((std / R) - 1))
    return threshold

# @profile
def load_and_preprocess_image(input_img_array):
    """
    이미지 배열을 받아서,
    1) BGR 이미지로 변환 (입력이 흑백이면 3채널로 변환)
    2) 그레이스케일 변환
    3) 가우시안 블러 적용
    4) Hessian 행렬 기반으로 ridge(극값) 검출
    5) 빠른 Sauvola thresholding (cv2.boxFilter 기반) 적용하여 binary 이미지 생성
    
    ※ Sauvola 알고리즘은 반드시 사용해야 하며, 이미지 다운샘플링은 하지 않습니다.
    """
    # 입력 이미지 차원에 따라 BGR 이미지 생성
    if input_img_array.ndim == 2:
        original_img = cv2.cvtColor(input_img_array, cv2.COLOR_GRAY2BGR)
    elif input_img_array.ndim == 3:
        original_img = input_img_array.copy()
    else:
        raise ValueError(f"Unexpected input dimensions: {input_img_array.ndim}")
    
    # 그레이스케일 변환 및 가우시안 블러 적용
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    
    # Hessian 행렬 기반으로 ridge 검출 (여기서 sigma=3.0 사용)
    _, b = detect_ridges(blurred_img, sigma=3.0)
    
    # Sauvola thresholding: 기존 skimage의 threshold_sauvola 대신 fast 구현 사용
    # cv2.boxFilter는 C++ 최적화 함수이므로 훨씬 빠르게 동작합니다.
    sauvola_thresh = sauvola_threshold_fast(b, window_size=15, k=0.5, R=128)
    binary_H_img = (b > sauvola_thresh).astype(np.uint8) * 255
    inverted_img = 255 - binary_H_img
    binary_img = inverted_img
    
    return original_img, gray_img, blurred_img, binary_img

def extract_joints(binary_img):
    """
    binary 이미지를 받아서
    수평/수직 마스크를 만들고, 교차점(centroid)들을 검출
    """
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))

    horizontal_mask = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_mask = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, vertical_kernel)

    joints_mask = cv2.bitwise_and(horizontal_mask, vertical_mask)
    contours, _ = cv2.findContours(joints_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centroids = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroids.append((cX, cY))

    return horizontal_mask, vertical_mask, centroids


def detect_largest_blob(original_img, binary_img, clipLimit):
    """
    SimpleBlobDetector로 블롭들을 검출한 뒤,
    각 블롭에 대해 확장(mask)하여 가장 큰 컨투어(=실린더 후보 영역)를 찾는다.
    
    skimage의 equalize_adapthist 대신 LAB 색 공간에서 L 채널에 CLAHE를 적용하여
    대비를 개선하는 방식으로 속도를 개선합니다.
    """
    # 컬러 이미지인 경우 LAB 색 공간으로 변환하여 L 채널에 CLAHE 적용
    if len(original_img.shape) == 3 and original_img.shape[2] == 3:
        lab = cv2.cvtColor(original_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        # CLAHE 적용 (tileGridSize와 clipLimit는 원하는 결과에 맞게 조정)
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(4,4))
        cl = clahe.apply(l)
        # CLAHE 처리된 L 채널을 이용해 LAB 이미지를 재구성하고, 필요시 BGR로 변환
        lab = cv2.merge((cl, a, b))
        equalized_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        # blob 검출에는 L 채널(그레이스케일 효과) 사용
        gray = cl
    else:
        # 입력 이미지가 이미 그레이스케일이면 바로 CLAHE 적용
        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8,8))
        gray = clahe.apply(original_img)
        equalized_img = gray

    # SimpleBlobDetector 설정
    params = cv2.SimpleBlobDetector_Params()
    params.minArea = 10
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False

    blob_detector = cv2.SimpleBlobDetector_create(params)
    keypoints = blob_detector.detect(gray)

    # 블롭 영역 확장을 위한 빈 이미지 생성 (equalized_img와 동일한 크기/채널)
    extended_blob_img = np.zeros_like(equalized_img)
    # 만약 equalized_img가 그레이스케일이면 3채널로 변환 (circle 그리기 용도)
    if len(extended_blob_img.shape) == 2:
        extended_blob_img = cv2.cvtColor(extended_blob_img, cv2.COLOR_GRAY2BGR)

    # 검출된 각 블롭에 대해 원을 그려 확장 영역 마스크 생성
    for keypoint in keypoints:
        x, y = keypoint.pt
        radius = keypoint.size / 2
        expanded_radius = int(radius + 4)
        cv2.circle(extended_blob_img, (int(x), int(y)), expanded_radius, (255, 255, 255), -1)

    # 확장 블롭 이미지를 그레이스케일로 변환 후 바이너리화하여 컨투어 검출
    extended_blob_img_gray = cv2.cvtColor(extended_blob_img, cv2.COLOR_BGR2GRAY)
    _, extended_blob_bin = cv2.threshold(extended_blob_img_gray, 1, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(extended_blob_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 가장 큰 컨투어(면적 기준) 선택
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    hull = cv2.convexHull(max_contour)
    mask_contour = np.zeros_like(binary_img)
    cv2.drawContours(mask_contour, [hull], -1, 255, thickness=-1)


    return max_contour, mask_contour

def get_convex_hull(img, threshold=127, expansion_pixels=20, visualize=False):
    """
    이미지로부터 볼록 껍질(컨투어)을 검출하고, 이를 바깥으로 지정된 픽셀만큼 확장하여 반환하는 함수.

    Parameters:
        img (numpy.ndarray): 처리할 이미지 배열 (그레이스케일 또는 컬러).
        threshold (int): 이진화 임계값 (기본값: 127).
        expansion_pixels (int): 볼록 껍질을 확장할 픽셀 수 (기본값: 20).
        visualize (bool): 결과를 시각화할지 여부 (기본값: False).

    Returns:
        expanded_hull (numpy.ndarray): 확장된 볼록 껍질을 이루는 점들의 배열. Shape: (n_points, 1, 2), dtype=int32.
    """
        
    # 이진화 (Thresholding)
    _, thresh = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    
    # 이진화된 이미지가 단일 채널인지 확인
    if len(thresh.shape) != 2:
        thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    
    # 컨투어 찾기
    contours_info = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_info[-2] if len(contours_info) == 3 else contours_info[0]
    
    if not contours:
        raise ValueError("컨투어를 찾을 수 없습니다.")
    
    # 가장 큰 컨투어 선택 (필요에 따라 다른 선택 기준 사용 가능)
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour
    
    if max_contour is None:
        raise ValueError("가장 큰 컨투어를 찾을 수 없습니다.")
    
    # 볼록 껍질 계산
    hull = cv2.convexHull(max_contour)
    
    # 볼록 껍질을 확장하기 위해 마스크 생성
    mask = np.zeros_like(thresh)
    cv2.polylines(mask, [hull], isClosed=True, color=255, thickness=1)
    cv2.fillPoly(mask, [hull], 255)
    
    # 확장을 위한 커널 생성 (원형 구조 요소 사용)
    kernel_size = expansion_pixels * 2 + 1  # 홀수 크기
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # 마스크 확장 (Dilation)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    
    # 확장된 마스크에서 새로운 컨투어 찾기
    contours_info_dilated = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_dilated = contours_info_dilated[-2] if len(contours_info_dilated) == 3 else contours_info_dilated[0]
    
    if not contours_dilated:
        raise ValueError("확장된 컨투어를 찾을 수 없습니다.")
    
    # 확장된 컨투어 중 가장 큰 컨투어 선택
    largest_contour_dilated = max(contours_dilated, key=cv2.contourArea)
    expanded_hull = cv2.convexHull(largest_contour_dilated)
    mask_contour = np.zeros_like(thresh)
    cv2.drawContours(mask_contour, [expanded_hull], -1, 255, thickness=-1)
    
    if visualize:
        # 원본 이미지 컬러로 변환 (시각화를 위해)
        if len(img.shape) == 2:
            img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_color = img.copy()
        
        # 원본 볼록 껍질 그리기 (빨간색)
        cv2.polylines(img_color, [hull], isClosed=True, color=(0, 0, 255), thickness=2)
        
        # 확장된 볼록 껍질 그리기 (파란색)
        cv2.polylines(img_color, [expanded_hull], isClosed=True, color=(255, 0, 0), thickness=2)
        
        # 시각화
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        if len(img.shape) == 2:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.title("Convex Hull and Expanded Hull")
        plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.show()
    
    return expanded_hull, mask_contour


def find_cylinder_centroids_and_center(centroids, max_contour, gray_img, original_img):
    """
    가장 큰 컨투어 영역(실린더 추정) 안에 존재하는 joint(centroid)들만 추려서,
    그 중 밝기가 가장 큰 점을 중심점(center_point)으로 결정
    """
    final_img = original_img.copy()
    cylinder_centroids = []
    center_point = None
    max_brightness = -1
    radius_for_second_nearest = 0

    if max_contour is not None:
        x, y, w, h = cv2.boundingRect(max_contour)
        cv2.rectangle(final_img, (x, y), (x + w, y + h), (0, 255, 255), 3)

        for (cX, cY) in centroids:
            if x <= cX < x + w and y <= cY < y + h:
                cv2.circle(final_img, (cX, cY), 3, (0, 0, 255), -1)
                cylinder_centroids.append((cX, cY))

                roi = gray_img[max(0, cY - 5):min(gray_img.shape[0], cY + 6),
                               max(0, cX - 5):min(gray_img.shape[1], cX + 6)]
                avg_brightness = np.mean(roi)
                if avg_brightness > max_brightness:
                    max_brightness = avg_brightness
                    center_point = (cX, cY)
            cv2.circle(final_img, center_point, 3, (0,255,0), -1)

        # center_point가 있고, blob(centroids) 2개 이상이면
        # 두 번째로 가까운 점과의 거리로부터 임의 radius를 구함
        if center_point is not None and len(cylinder_centroids) >= 2:
            distances = []
            for blob in cylinder_centroids:
                distance = math.hypot(center_point[0] - blob[0], center_point[1] - blob[1])
                distances.append((distance, blob))
            distances.sort()
            second_min_distance, second_nearest_blob = distances[1]
            radius_for_second_nearest = int(second_min_distance)

    return final_img, cylinder_centroids, center_point, radius_for_second_nearest

def mask_roi_around_center(horizontal_mask, vertical_mask,
                           mask_contour, original_img):
    """
    기존 (center_point, radius)를 사용하여 ROI 영역을 제거하는 방식을 대신하여,
    original_img에 GaussianBlur((23,23))와 threshold(235)를 적용한 후,
    이진화된 이미지에서 가장 큰 영역의 최소 원을 구합니다.
    구해진 원 내부 영역을 horizontal_mask와 vertical_mask에서 제거(0 설정)하여
    실린더 내부 영역을 보정합니다.
    
    참고: 매개변수 max_contour, center_point, radius는 새 알고리즘에서 사용하지 않습니다.
    """
    # 1. 원본 이미지(original_img)를 그레이스케일로 변환 (이미 그레이스케일이면 그대로 사용)
    if len(original_img.shape) > 2:
        image_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = original_img.copy()
    
    # 2. GaussianBlur 적용 (커널 크기 23x23)
    blurred = cv2.GaussianBlur(image_gray, (19, 19), 0)
    
    # 3. 밝기 기준 235로 threshold 적용하여 이진화
    _, binary_image = cv2.threshold(blurred, 240, 240, cv2.THRESH_BINARY)
    
    # 4. 이진화된 이미지에서 외부 contour들 찾기
    contours, _ = cv2.findContours(binary_image.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 5. 원의 영역을 제거하기 위한 마스크 생성 (전체 흰색: 255)
    circle_mask = np.ones(horizontal_mask.shape, dtype=np.uint8) * 255
    
    if contours:
        # 가장 큰 contour 선택
        largest_contour = max(contours, key=cv2.contourArea)
        # 최소 원 계산: (x, y)는 원의 중심, circle_radius는 반지름
        (x, y), circle_radius = cv2.minEnclosingCircle(largest_contour)
        circle_center = (int(x), int(y))
        circle_radius = int(circle_radius)
        
        if circle_radius < 30:
            circle_radius = circle_radius + 0
        else:
            circle_radius = circle_radius + 0
        
        # circle_mask에서 해당 원 영역을 0으로 채움 (원 내부 제거)
        cv2.circle(circle_mask, circle_center, (circle_radius), 0, thickness=-1)
    
    # 6. 수평/수직 마스크에 원 제거 마스크 적용 (비트연산)
    mask_roi_h = cv2.bitwise_and(horizontal_mask, circle_mask)
    mask_roi_v = cv2.bitwise_and(vertical_mask, circle_mask)
    
    # 7. 추가적으로 mask_contour 적용
    mask_roi_h = cv2.bitwise_and(mask_roi_h, mask_contour)
    mask_roi_v = cv2.bitwise_and(mask_roi_v, mask_contour)
    
    # 8. 모폴로지 연산 (오프닝)으로 잔여 노이즈 제거
    kernel = np.ones((3, 3), np.uint8)
    mask_roi_h = cv2.morphologyEx(mask_roi_h, cv2.MORPH_OPEN, kernel)
    mask_roi_v = cv2.morphologyEx(mask_roi_v, cv2.MORPH_OPEN, kernel)
    
    return mask_roi_h, mask_roi_v, circle_radius


# kernprof -v -l python_grid_detection_cylinder.py
# python -m line_profiler python_grid_detection_cylinder.py.lprof > results_main.txt

# @profile
def color_and_expand_lines(mask_roi_h, mask_roi_v, circle_radius, max_contour, mask_contour, original_img, cylinder_centroids):
    """
    마스크를 라벨링하여 색상화, 라인 확장(arc fitting)까지 수행
    """
    if max_contour is not None:
        x, y, w, h = cv2.boundingRect(max_contour)

        # # rows, cols 끊어진 roi 확장해서 잇기
        horizontal_expanded, horizontal_dubug = expands_line_roi(mask_roi_h, 1, mask_contour, kernel_size=201)
        vertical_expanded, vertical_dubug = expands_line_roi(mask_roi_v, 1, mask_contour, kernel_size=201)

        # rows, cols label 설정(row1, row2, ,,,) 준비
        distinct_colors = generate_distinct_colors(20)
        num_labels_h2, labels_h2, horizontal_colored2 = label_and_color_masks(horizontal_expanded[y:y+h, x:x+w],generate_unique_colors(20, distinct_colors))
        num_labels_v2, labels_v2, vertical_colored2 = label_and_color_masks(vertical_expanded[y:y+h, x:x+w],generate_unique_colors(20, distinct_colors))

        # label 설정
        rows = group_points_by_label(cylinder_centroids, labels_h2, x, y)
        cols = group_points_by_label(cylinder_centroids, labels_v2, x, y)

        # rows, cols ['equations'] 초기화 (degree=2)
        rows, cols = create_dummy_rows_cols(rows, cols, degree=1)

        # rows, cols ['points'] 에 degree=3인 poly fitting => 비정상 label 제거 => fitted poly간 intesrection point 구하기
        img_with_poly, rows, cols = fit_and_draw_polynomial(original_img, rows, cols, w, h, max_contour, degree=1)        
        # rows, cols = remove_label(rows, cols)
        # img_with_poly_G, rows_G, cols_G = modify_grayscale_Cline(original_img, rows, cols, draw_points=True, degree=degree, window_size=7)             
        img_with_points, rows_updated, cols_updated = find_and_assign_intersections_P(img_with_poly, rows, cols, max_contour, draw_points=True, degree=1)

        # 비어있는 label 제거 및 label이 1부터 시작하게끔 re-naming
        rows_updated, cols_updated = clean_and_relabel(rows_updated, cols_updated)
        
        # center point 구하고 center에 맞춰서 row, col indexing
        result_json, result_dict, rows_dict, cols_dict, center_point = indexing_data(
            rows_updated, cols_updated, original_img, mask_contour, circle_radius
        )

        # center point, row, cols ['point'] 시각화
        row_img, col_img = draw_points(original_img, rows_dict, cols_dict, center_point)

        # matlab으로 넘기기 위해 json 변환
        result_json = make_json(center_point, cols_dict)
        
        return col_img, result_json, rows_updated, cols_updated

    # max_contour가 None인 경우, 그냥 결과 없는 값
    return original_img, json.dumps({}), {}, {}