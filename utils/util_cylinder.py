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
       dilation을 수행한 결과(확장 마스크와 디버그 이미지를)를 반환.
       
       단, 해당 컴포넌트의 PCA 길이(length)가 0.8×global_length보다 큰 경우에는
       확장을 수행하지 않습니다.
       
    반환:
        (expanded_contribution, debug_contribution)
            expanded_contribution: 해당 컨투어의 확장 결과 (base_mask와 동일 크기의 binary mask)
            debug_contribution: 해당 컨투어의 확장 영역을 연두색(BGR: (144,238,144))으로 표시한 이미지 (3채널)
    """
    expanded_contribution = np.zeros_like(base_mask)
    debug_contribution = np.zeros((h, w, 3), dtype=np.uint8)
    p1, p2, angle = info['p1'], info['p2'], info['angle']
    comp_length = info.get('length', None)
    # angle, p1, p2가 None이거나 PCA 길이가 0.8×global_length보다 크면 확장하지 않음.
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

def expand_line_roi(mask_roi, patch_size=15, kernel_size=81, min_pixels=5, max_pixels=200):
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

def expand_line_roi_before(mask_roi, endpoints_dict, direction='horizontal', expand_pixels=10):
    """
    Expand line ROI by 3 pixels at a time.
    - If connected component count decreases => extra 3-pixel expand & stop for that label.
    - After finishing each label's expansion, update the 'original_num_labels' to the new count.
      Thus, each label uses the most recent label count as baseline.

    Parameters
    ----------
    mask_roi : np.ndarray
        (H,W) uint8. 0=background, n=object or 255=object
    endpoints_dict : dict
        { label_val: (minX, maxX, minY, maxY, points) }
    direction : str
        'horizontal' or 'vertical'
    expand_pixels : int
        up to how many pixels we expand in total (default=10)

    Returns
    -------
    expanded_mask : np.ndarray
        (0/255) final expanded mask
    """

    # 1) prepare initial
    expanded_mask = np.where(mask_roi>0, 255, 0).astype(np.uint8)

    # initial connected components
    original_num_labels, _ = cv2.connectedComponents(expanded_mask, connectivity=8)

    thickness = 3
    half_thickness = thickness/2.0
    edge_margin = 5

    # 2) iterate over each label
    for label_val, (minX, maxX, minY, maxY, points) in endpoints_dict.items():
        pts = np.array(points, dtype=np.float32)
        if len(pts)<2:
            continue

        # find endpoints
        if direction=='horizontal':
            left_region  = pts[(pts[:,0]>=minX)&(pts[:,0]<=minX+edge_margin)]
            right_region = pts[(pts[:,0]>=maxX-edge_margin)&(pts[:,0]<=maxX)]
            if len(left_region)==0 or len(right_region)==0:
                continue
            x1, y1 = left_region[:,0].mean(),  left_region[:,1].mean()
            x2, y2 = right_region[:,0].mean(), right_region[:,1].mean()
        else:
            top_region    = pts[(pts[:,1]>=minY)&(pts[:,1]<=minY+edge_margin)]
            bottom_region = pts[(pts[:,1]>=maxY-edge_margin)&(pts[:,1]<=maxY)]
            if len(top_region)==0 or len(bottom_region)==0:
                continue
            x1, y1 = top_region[:,0].mean(),   top_region[:,1].mean()
            x2, y2 = bottom_region[:,0].mean(),bottom_region[:,1].mean()

        dx, dy = x2 - x1, y2 - y1
        length = np.hypot(dx, dy)
        if length<1e-8:
            continue

        ux, uy = dx/length, dy/length
        px, py = -uy, ux

        # # === (A) start point (opposite direction) ===
        # move_dist = 3
        # while move_dist<=expand_pixels:
        #     x1_ex = x1 - move_dist*ux
        #     y1_ex = y1 - move_dist*uy

        #     c1 = (x1_ex + px*half_thickness, y1_ex + py*half_thickness)
        #     c2 = (x1_ex - px*half_thickness, y1_ex - py*half_thickness)
        #     c3 = (x1    - px*half_thickness, y1    - py*half_thickness)
        #     c4 = (x1    + px*half_thickness, y1    + py*half_thickness)
        #     poly = np.array([c1,c2,c3,c4], dtype=np.int32)

        #     cv2.fillPoly(expanded_mask, [poly], 255)

        #     num_labels_now, _ = cv2.connectedComponents(expanded_mask, connectivity=8)
        #     if num_labels_now < original_num_labels:
        #         # merge => extra +3
        #         extra_dist = move_dist+3
        #         if extra_dist<=expand_pixels:
        #             x1_ex2 = x1 - extra_dist*ux
        #             y1_ex2 = y1 - extra_dist*uy

        #             c1_2 = (x1_ex2 + px*half_thickness, y1_ex2 + py*half_thickness)
        #             c2_2 = (x1_ex2 - px*half_thickness, y1_ex2 - py*half_thickness)
        #             c3_2 = (x1     - px*half_thickness, y1     - py*half_thickness)
        #             c4_2 = (x1     + px*half_thickness, y1     + py*half_thickness)
        #             poly2 = np.array([c1_2,c2_2,c3_2,c4_2], dtype=np.int32)
        #             cv2.fillPoly(expanded_mask, [poly2], 255)

        #         break
        #     else:
        #         move_dist += 3

        # === (B) end point (forward direction) ===
        move_dist = 3
        while move_dist<=expand_pixels:
            x2_ex = x2 + move_dist*ux
            y2_ex = y2 + move_dist*uy

            c1 = (x2_ex - px*half_thickness, y2_ex - py*half_thickness)
            c2 = (x2_ex + px*half_thickness, y2_ex + py*half_thickness)
            c3 = (x2    + px*half_thickness, y2    + py*half_thickness)
            c4 = (x2    - px*half_thickness, y2    - py*half_thickness)
            poly = np.array([c1,c2,c3,c4], dtype=np.int32)

            cv2.fillPoly(expanded_mask, [poly], 255)

            num_labels_now, _ = cv2.connectedComponents(expanded_mask, connectivity=8)
            if num_labels_now < original_num_labels:
                # merge => extra +3
                extra_dist = move_dist+3
                if extra_dist<=expand_pixels:
                    x2_ex2 = x2 + extra_dist*ux
                    y2_ex2 = y2 + extra_dist*uy

                    c1_2 = (x2_ex2 - px*half_thickness, y2_ex2 - py*half_thickness)
                    c2_2 = (x2_ex2 + px*half_thickness, y2_ex2 + py*half_thickness)
                    c3_2 = (x2    + px*half_thickness,  y2    + py*half_thickness)
                    c4_2 = (x2    - px*half_thickness,  y2    - py*half_thickness)
                    poly2 = np.array([c1_2,c2_2,c3_2,c4_2], dtype=np.int32)
                    cv2.fillPoly(expanded_mask, [poly2], 255)

                break
            else:
                move_dist += 3

        # === (C) 한 라벨 확장 끝난 후, num_labels 업데이트 ===
        new_num_labels, _ = cv2.connectedComponents(expanded_mask, connectivity=8)
        original_num_labels = new_num_labels

    return expanded_mask


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

# Function to sort rows by y-coordinate
def sort_rows(points_grouped):
    sorted_rows = sorted(points_grouped.items(), key=lambda item: min(point[1] for point in item[1]))
    return sorted_rows

# Function to sort columns by x-coordinate
def sort_cols(points_grouped):
    sorted_cols = sorted(points_grouped.items(), key=lambda item: min(point[0] for point in item[1]))
    return sorted_cols

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
    행(row): y = f(x)
    열(col): x = f(y)
    각각 n차 다항식으로 피팅하여 이미지에 그리며, equations에 저장합니다.
    (열에 대해서는 비정상 열 병합 없이 행 처리와 동일한 방식으로 진행됩니다.)
    """
    img_with_poly = img.copy()

    # (B) 열에 대한 처리: 행 처리와 동일한 방식으로 진행 (비정상 열 병합 로직 제거)
    for col_name, points in cols["points"].items():
        if len(points) < degree + 1:
            # 피팅을 위해 최소 degree+1개의 점 필요
            continue

        pts = np.array(points, dtype=np.float32)
        # y 기준 정렬 (열은 x = f(y) 관계)
        pts = pts[np.argsort(pts[:, 1])]

        y_vals = pts[:, 1]
        x_vals = pts[:, 0]

        # 다항식 피팅 (x = f(y))
        poly_coeff = polynomial_fitting_col(y_vals, x_vals, degree=degree)

        # 도메인 설정 (행과 동일하게 50씩 확장)
        y_min, y_max = y_vals.min(), y_vals.max()
        y_min = y_min - 50
        y_max = y_max + 50

        # equations에 저장 (계수 + [y_min, y_max, abs(y_max - y_min)])
        cols["equations"][col_name] = list(poly_coeff) + [float(y_min), float(y_max), abs(float(y_max)-float(y_min))]

        # 곡선 샘플링 및 그리기
        num_points = max(50, len(y_vals))
        y_samp = np.linspace(y_min, y_max, num_points)
        x_samp = np.polyval(poly_coeff, y_samp)

        for i in range(num_points - 1):
            pt1 = (int(x_samp[i]), int(y_samp[i]))
            pt2 = (int(x_samp[i+1]), int(y_samp[i+1]))
            cv2.line(img_with_poly, pt1, pt2, (255, 0, 0), 1)  # 파란색

    # (A) 행에 대한 처리: 기존 로직 그대로 진행
    for row_name, points in rows["points"].items():
        if len(points) < degree + 1:
            # 피팅을 위해 최소 degree+1개의 점 필요
            continue

        pts = np.array(points, dtype=np.float32)
        # x 기준 정렬 (행은 y = f(x) 관계)
        pts = pts[np.argsort(pts[:, 0])]

        x_vals = pts[:, 0]
        y_vals = pts[:, 1]

        # 다항식 피팅 (y = f(x))
        poly_coeff = polynomial_fitting_row(x_vals, y_vals, degree=degree)

        # 도메인 설정 (x 축 확장)
        x_min, x_max = x_vals.min(), x_vals.max()
        x_min = x_min - 50
        x_max = x_max + 50

        # equations에 저장 (계수 + [x_min, x_max, abs(x_max - x_min)])
        rows["equations"][row_name] = list(poly_coeff) + [float(x_min), float(x_max), abs(float(x_max)-float(x_min))]

        # 곡선 샘플링 및 그리기
        num_points = max(50, len(x_vals))
        x_samp = np.linspace(x_min, x_max, num_points)
        y_samp = np.polyval(poly_coeff, x_samp)

        for i in range(num_points - 1):
            pt1 = (int(x_samp[i]), int(y_samp[i]))
            pt2 = (int(x_samp[i+1]), int(y_samp[i+1]))
            cv2.line(img_with_poly, pt1, pt2, (0, 255, 0), 1)  # 녹색

    return img_with_poly, rows, cols

def fit_and_draw_polynomial2(img, rows, cols, max_w, max_h, max_contour, degree=3):
    """
    행(row): y = f(x)
    열(col): x = f(y)
    각각 n차 다항식으로 피팅하여 이미지에 그리며, equations에 저장합니다.
    
    열의 경우, 각 그래프를 이미지 끝까지 확장했을 때,
    cv2.boundingRect(max_contour)를 통해 얻은 중간 가로선( y_mid = y + h/2 )과
    다항식 곡선이 만나는 점의 x좌표(x_intersect)를 기준으로,
    두 열의 x_intersect 차이가 3픽셀 이하이면 병합합니다.
    """
    import numpy as np
    import cv2

    img_with_poly = img.copy()

    # max_contour로부터 bounding rectangle 계산 및 중간 가로선 y 좌표 구하기
    x_b, y_b, w, h = cv2.boundingRect(max_contour)
    y_mid = y_b + h/2

    # -----------------------------------------
    # (B) 열에 대한 처리: 각 열에 대해 다항식 피팅 및 특징 추출
    # -----------------------------------------
    col_fits = {}  
    for col_name, points in cols["points"].items():
        if len(points) < degree + 1:
            continue

        pts = np.array(points, dtype=np.float32)
        # y 기준 정렬 (열: x = f(y))
        pts = pts[np.argsort(pts[:, 1])]
        y_vals = pts[:, 1]
        x_vals = pts[:, 0]

        # 다항식 피팅 (x = f(y))
        poly_coeff = polynomial_fitting_col(y_vals, x_vals, degree=degree)

        # 도메인 설정: y값 최소/최대에 50픽셀씩 확장
        y_min, y_max = y_vals.min(), y_vals.max()
        y_min_ext = y_min - 50
        y_max_ext = y_max + 50

        # 임시로 equations에 저장 (계수와 y 범위 정보)
        cols["equations"][col_name] = list(poly_coeff) + [float(y_min_ext), float(y_max_ext),
                                                           abs(float(y_max_ext)-float(y_min_ext))]

        # 그래프를 이미지 끝까지 확장하여 샘플링
        num_points = max(50, len(y_vals))
        y_samp = np.linspace(y_min_ext, y_max_ext, num_points)
        x_samp = np.polyval(poly_coeff, y_samp)

        # **중간 가로선(y = y_mid)과 만나는 x 좌표 계산**
        x_intersect = np.polyval(poly_coeff, y_mid)

        # 특징 벡터: 여기서는 x_intersect만 사용
        feature = [x_intersect]

        col_fits[col_name] = {"points": points, "equation": poly_coeff,
                              "y_range": (y_min_ext, y_max_ext),
                              "x_intersect": x_intersect,
                              "y_samp": y_samp, "x_samp": x_samp}
        # 개별 열 곡선은 그리지 않음.

    # -----------------------------------------
    # 열 병합: 각 열의 x_intersect 차이가 3픽셀 이하이면 병합
    # -----------------------------------------
    merged_cols = {}        # 병합된 points 저장용
    merged_equations = {}   # 병합된 equations 저장용

    # x_intersect 기준으로 정렬
    sorted_labels = sorted(col_fits.keys(), key=lambda l: col_fits[l]["x_intersect"])
    processed = set()
    group_id = 0
    for label in sorted_labels:
        if label in processed:
            continue
        group = [label]
        processed.add(label)
        for other in sorted_labels:
            if other in processed:
                continue
            if abs(col_fits[other]["x_intersect"] - col_fits[label]["x_intersect"]) <= 10:
                group.append(other)
                processed.add(other)
        # 그룹에 속한 열들의 points 병합 (리스트 단순 연결)
        merged_points_list = []
        for l in group:
            merged_points_list.extend(col_fits[l]["points"])
        merged_points = merged_points_list

        # 재피팅: 병합된 points에 대해 다항식 피팅 (x = f(y))
        merged_pts = np.array(merged_points, dtype=np.float32)
        if len(merged_pts) < degree + 1:
            continue
        merged_pts = merged_pts[np.argsort(merged_pts[:, 1])]  # y 기준 정렬
        y_vals = merged_pts[:, 1]
        x_vals = merged_pts[:, 0]
        new_poly_coeff = polynomial_fitting_col(y_vals, x_vals, degree=degree)
        y_min_new, y_max_new = y_vals.min(), y_vals.max()
        y_min_new_ext = y_min_new - 50
        y_max_new_ext = y_max_new + 50

        new_label = f"merged_{group_id}"
        group_id += 1

        merged_cols[new_label] = merged_points  # points 병합
        merged_equations[new_label] = list(new_poly_coeff) + [float(y_min_new_ext), float(y_max_new_ext),
                                                               abs(float(y_max_new_ext)-float(y_min_new_ext))]
    
        # 병합된 열의 곡선 그리기 (파란색)
        num_points = max(50, len(y_vals))
        y_samp = np.linspace(y_min_new_ext, y_max_new_ext, num_points)
        x_samp = np.polyval(new_poly_coeff, y_samp)
        for i in range(num_points - 1):
            pt1 = (int(x_samp[i]), int(y_samp[i]))
            pt2 = (int(x_samp[i+1]), int(y_samp[i+1]))
            cv2.line(img_with_poly, pt1, pt2, (255, 0, 0), 1)
    
    # 병합된 결과로 cols["points"]와 cols["equations"] 업데이트 (개별 열은 버림)
    cols["points"] = merged_cols
    cols["equations"] = merged_equations

    # -----------------------------------------
    # (A) 행에 대한 처리: 기존 로직 그대로 진행
    # -----------------------------------------
    for row_name, points in rows["points"].items():
        if len(points) < degree + 1:
            continue

        pts = np.array(points, dtype=np.float32)
        pts = pts[np.argsort(pts[:, 0])]  # x 기준 정렬 (행: y = f(x))
        x_vals = pts[:, 0]
        y_vals = pts[:, 1]

        # 다항식 피팅 (y = f(x))
        poly_coeff = polynomial_fitting_row(x_vals, y_vals, degree=degree)
        x_min, x_max = x_vals.min(), x_vals.max()
        x_min_ext = x_min - 50
        x_max_ext = x_max + 50

        rows["equations"][row_name] = list(poly_coeff) + [float(x_min_ext), float(x_max_ext),
                                                           abs(float(x_max_ext)-float(x_min_ext))]
        num_points = max(50, len(x_vals))
        x_samp = np.linspace(x_min_ext, x_max_ext, num_points)
        y_samp = np.polyval(poly_coeff, x_samp)
        for i in range(num_points - 1):
            pt1 = (int(x_samp[i]), int(y_samp[i]))
            pt2 = (int(x_samp[i+1]), int(y_samp[i+1]))
            cv2.line(img_with_poly, pt1, pt2, (0, 255, 0), 1)  # 녹색

    return img_with_poly, rows, cols



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


def process_row(label, eq, gray_float, degree, sample_step, window_size, draw_points):
    """
    단일 row label에 대한 처리 함수.
    eq: [a, b, x_min, x_max, ...] (1차 기준)
    반환: (label, new_eq, drawing_segments)
    drawing_segments: 각 선분을 (pt1, pt2, color) 튜플로 반환 (draw_points가 True일 경우)
    """
    drawing_segments = []
    # 최소한 1차 다항식이면 5개 이상의 값이 필요함
    if len(eq) < degree + 4:
        return (label, None, drawing_segments)
    
    poly_coeff = eq[:degree+1]  # 예: [a, b]
    x_min = eq[degree+1]
    x_max = eq[degree+2]
    if x_max < x_min:
        return (label, None, drawing_segments)
    
    # 1) (x_min ~ x_max) 구간에서 점 샘플링 (x 좌표 기준)
    sampled_points = []
    x_vals = np.arange(x_min, x_max + 0.0001, sample_step)
    if degree == 1:
        a, b = poly_coeff
        for xv in x_vals:
            yv = a * xv + b
            sampled_points.append((xv, yv))
    else:
        p = np.poly1d(poly_coeff)
        for xv in x_vals:
            yv = p(xv)
            sampled_points.append((xv, yv))
    
    # 2) y 좌표에 대해 중력중심 보정 (compute_center_of_gravity_y 함수 사용)
    refined = compute_center_of_gravity_y(gray_float, sampled_points, window_size=window_size)
    rx = refined[:, 0]
    ry = refined[:, 1]
    if len(rx) < degree + 1:
        return (label, None, drawing_segments)
    
    # 3) 보정된 점으로 다시 다항식 피팅 (x -> y)
    new_coefs = np.polyfit(rx, ry, degree)
    new_x_min, new_x_max = float(np.min(rx)), float(np.max(rx))
    abs_diff = abs(new_x_max - new_x_min)
    new_eq = list(new_coefs) + [new_x_min, new_x_max, abs_diff]
    
    # 4) draw_points=True이면, 샘플링한 점들로 선분 좌표 생성 (초록색)
    if draw_points:
        num_draw = max(100, len(rx))
        x_samp = np.linspace(new_x_min, new_x_max, num_draw)
        y_samp = np.polyval(new_coefs, x_samp)
        for i in range(num_draw - 1):
            pt1 = (int(round(x_samp[i])), int(round(y_samp[i])))
            pt2 = (int(round(x_samp[i+1])), int(round(y_samp[i+1])))
            drawing_segments.append((pt1, pt2, (0, 255, 0)))  # green
    return (label, new_eq, drawing_segments)

def process_col(label, eq, gray_float, degree, sample_step, window_size, draw_points):
    """
    단일 col label에 대한 처리 함수.
    eq: [a, b, y_min, y_max, ...] (1차 기준)
    반환: (label, new_eq, drawing_segments)
    drawing_segments: (pt1, pt2, color) 튜플 리스트 (파란색)
    """
    drawing_segments = []
    if len(eq) < degree + 4:
        return (label, None, drawing_segments)
    
    poly_coeff = eq[:degree+1]  # 예: [a, b]
    y_min = eq[degree+1]
    y_max = eq[degree+2]
    if y_max < y_min:
        return (label, None, drawing_segments)
    
    sampled_points = []
    y_vals = np.arange(y_min, y_max + 0.0001, sample_step)
    if degree == 1:
        a, b = poly_coeff
        for yv in y_vals:
            xv = a * yv + b
            sampled_points.append((xv, yv))
    else:
        p = np.poly1d(poly_coeff)
        for yv in y_vals:
            xv = p(yv)
            sampled_points.append((xv, yv))
    
    # x좌표에 대해 중력중심 보정
    refined = compute_center_of_gravity_x(gray_float, sampled_points, window_size=window_size)
    rx = refined[:, 0]
    ry = refined[:, 1]
    if len(ry) < degree + 1:
        return (label, None, drawing_segments)
    
    new_coefs = np.polyfit(ry, rx, degree)
    new_y_min, new_y_max = float(np.min(ry)), float(np.max(ry))
    abs_diff = abs(new_y_max - new_y_min)
    new_eq = list(new_coefs) + [new_y_min, new_y_max, abs_diff]
    
    if draw_points:
        num_draw = 100
        y_samp = np.linspace(new_y_min, new_y_max, num_draw)
        x_samp = np.polyval(new_coefs, y_samp)
        for i in range(num_draw - 1):
            pt1 = (int(round(x_samp[i])), int(round(y_samp[i])))
            pt2 = (int(round(x_samp[i+1])), int(round(y_samp[i+1])))
            drawing_segments.append((pt1, pt2, (255, 0, 0)))  # blue
    return (label, new_eq, drawing_segments)

def modify_grayscale_Cline(
    input_img, rows, cols, 
    draw_points=True, 
    degree=1,
    sample_step=1.0,
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

    gray_float = img_as_float(gray_img)
    rows_updated = copy.deepcopy(rows)
    cols_updated = copy.deepcopy(cols)
    
    row_drawing_segments = []
    col_drawing_segments = []
    
    # (A) Rows 병렬 처리
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_rows = []
        for label, eq in rows_updated.get("equations", {}).items():
            future = executor.submit(process_row, label, eq, gray_float, degree, sample_step, window_size, draw_points)
            future_rows.append(future)
        for future in concurrent.futures.as_completed(future_rows):
            label, new_eq, segments = future.result()
            if new_eq is not None:
                rows_updated["equations"][label] = new_eq
            row_drawing_segments.extend(segments)
    
    # (B) Cols 병렬 처리
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_cols = []
        for label, eq in cols_updated.get("equations", {}).items():
            future = executor.submit(process_col, label, eq, gray_float, degree, sample_step, window_size, draw_points)
            future_cols.append(future)
        for future in concurrent.futures.as_completed(future_cols):
            label, new_eq, segments = future.result()
            if new_eq is not None:
                cols_updated["equations"][label] = new_eq
            col_drawing_segments.extend(segments)
    
    # draw_points가 True이면, 메인 스레드에서 선분들을 그립니다.
    if draw_points and out_img is not None:
        for pt1, pt2, color in row_drawing_segments:
            cv2.line(out_img, pt1, pt2, color, 1)
        for pt1, pt2, color in col_drawing_segments:
            cv2.line(out_img, pt1, pt2, color, 1)
    
    if draw_points:
        return out_img, rows_updated, cols_updated
    else:
        return None, rows_updated, cols_updated

def merge_short_lines(line_dict, lengths_dict, max_len, ratio=0.7, is_row=True):
    """
    line_dict: rows 또는 cols ({"points": {...}, "equations": {...}})
    lengths_dict: compute_line_lengths() 결과
    max_len: 해당 line_dict 내에서의 최댓값
    ratio: 최대 길이 대비 몇 % 미만이면 '끊긴 선'으로 간주
    is_row: True면 row(가로), False면 col(세로)

    병합 로직 개념:
      - '끊긴 선' candidates를 찾아
      - candidates 각각에 대해, 다항식 '확장'해서 근방 다른 라벨과 만나는지 검사
      - 만나면 두 라벨의 points를 합침 -> 하나의 라벨로 병합
      - 최종 line_dict를 업데이트

    return: 병합 후 line_dict
    """
    if max_len <= 0:
        return line_dict  # 길이가 0인 경우 처리X

    # (1) '끊긴선' 후보 찾기
    threshold_len = max_len * ratio
    short_labels = [lbl for lbl, ln in lengths_dict.items() if ln < threshold_len]

    if not short_labels:
        return line_dict  # 짧은 선이 없으면 그대로 반환

    # (2) 실제 병합
    #     예시: short_labels의 각 라벨 vs 다른 라벨과 "근접"하면 병합
    #     병합 기준: '다항식 확장'해서 만나면 -> 간단 예시는 bounding box 근접 정도로 대체

    updated_points = dict(line_dict["points"])  # 수정 용도 복사
    merged_labels = set()  # 이미 병합된 라벨 기록

    for s_label in short_labels:
        if s_label in merged_labels:
            continue

        spoints = updated_points.get(s_label, [])
        if len(spoints) < 2:
            continue

        # s_label의 bounding box
        sp = np.array(spoints, dtype=np.float32)
        x_min_s, x_max_s = sp[:,0].min(), sp[:,0].max()
        y_min_s, y_max_s = sp[:,1].min(), sp[:,1].max()

        # 간단히: row면 x범위를 확장, col이면 y범위를 확장, 다른 라벨과 교차하는지 검사
        # *** 실제론 polynomial extension으로 교차점 찾는 로직이 필요하지만,
        #     예시에선 bounding box 근사로 대체함. ***

        for other_label, other_points in updated_points.items():
            if other_label == s_label:
                continue
            if other_label in merged_labels:
                continue
            if len(other_points) < 2:
                continue

            op = np.array(other_points, dtype=np.float32)
            x_min_o, x_max_o = op[:,0].min(), op[:,0].max()
            y_min_o, y_max_o = op[:,1].min(), op[:,1].max()

            # row일 때: x범위가 어느정도 겹치면 병합
            # col일 때: y범위가 어느정도 겹치면 병합
            if is_row:
                # 간단히 x-interval 교집합 존재하면 '만났다'고 가정
                if (x_max_s >= x_min_o) and (x_max_o >= x_min_s):
                    # 병합
                    merged_points = spoints + other_points
                    updated_points[s_label] = merged_points
                    updated_points[other_label] = []
                    merged_labels.add(other_label)
            else:
                # col인 경우 y-interval 교집합
                if (y_max_s >= y_min_o) and (y_max_o >= y_min_s):
                    # 병합
                    merged_points = spoints + other_points
                    updated_points[s_label] = merged_points
                    updated_points[other_label] = []
                    merged_labels.add(other_label)

    # (3) 정리: 병합된 라벨(other_label)은 points가 빈 리스트가 되었으므로 제거
    final_points = {}
    for lbl, pts in updated_points.items():
        if lbl in merged_labels and len(pts) == 0:
            # 이 라벨은 이미 병합되어 소멸
            continue
        if len(pts) < 1:
            continue
        final_points[lbl] = pts

    # (4) 갱신
    line_dict["points"] = final_points
    # 'equations'는 재피팅할 것이므로 여기서 일단 비워도 됨
    for lbl in line_dict["equations"].keys():
        if lbl not in final_points:
            line_dict["equations"][lbl] = []
    return line_dict



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


def clean_and_relabel(rows, cols):
    # 공통 처리 함수: data 내의 'points'와 'equations'를 함께 정렬 및 재라벨링
    # sort_axis: rows의 경우 1 (y좌표), cols의 경우 0 (x좌표)
    def process_side(data, prefix, sort_axis):
        points = data.get('points', {})
        equations = data.get('equations', {})
        
        # points가 dict가 아닌 경우 경고 출력 후 빈 dict 반환
        if not isinstance(points, dict):
            print(f"Warning: 'points' in {prefix} is not a dictionary.")
            return {}, {}
        
        # 빈 값이 아닌 것들만 필터링
        filtered_points = {label: pts for label, pts in points.items() if pts}
        
        # 각 라벨별로 처음 10개 점의 해당 좌표의 평균을 계산
        avg_dict = {}
        for label, pts in filtered_points.items():
            # pts가 리스트라 가정하고, 각 점은 (x,y) 또는 [x,y] 형식
            # 처음 10개 점만 사용
            selected_pts = pts
            coords = [pt[sort_axis] for pt in selected_pts if isinstance(pt, (list, tuple)) and len(pt) >= 2]
            if coords:
                avg_dict[label] = np.mean(coords)
            else:
                avg_dict[label] = float('inf')
        
        # 평균값 기준으로 오름차순 정렬한 라벨 리스트
        sorted_labels = sorted(filtered_points.keys(), key=lambda l: avg_dict[l])
        
        # 정렬된 순서대로 새로운 라벨 할당 (points와 equations 모두)
        new_points = {}
        new_equations = {}
        for i, old_label in enumerate(sorted_labels, start=1):
            new_label = f"{prefix}{i}"
            new_points[new_label] = filtered_points[old_label]
            # equations도 동일하게 처리 (단, 해당 라벨의 equation 값이 [0, 0, 0, 0]이면 제외)
            if old_label in equations and equations[old_label] != [0, 0, 0, 0]:
                new_equations[new_label] = equations[old_label]
        
        return new_points, new_equations

    # rows: y 좌표 기준 정렬 (sort_axis=1)
    new_row_points, new_row_equations = process_side(rows, 'row', 1)
    rows['points'] = new_row_points
    rows['equations'] = new_row_equations

    # cols: x 좌표 기준 정렬 (sort_axis=0)
    new_col_points, new_col_equations = process_side(cols, 'col', 0)
    cols['points'] = new_col_points
    cols['equations'] = new_col_equations

    return rows, cols




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

def remove_label_eq(rows, cols, threshold = 1e-5):
    """
    rows, cols 딕셔너리에서 'equations' 및 'points'에 저장된 방정식을
    각각 처리합니다.
    
    - rows의 각 방정식은 [a, b, c, xmin, xmax, ...] 형식이며,
      x에 대한 2차 함수로 간주하여, 허용 범위 [xmin, xmax] 내 최대 곡률을 계산합니다.
    - cols의 각 방정식은 [a, b, c, ymin, ymax, ...] 형식이며,
      y에 대한 2차 함수로 간주하여, 허용 범위 [ymin, ymax] 내 최대 곡률을 계산합니다.
      
    만약 최대 곡률이 1e-4 미만이면 해당 레이블을 제거한 후,
    남은 레이블들을 원래 순서대로 "row1", "row2", ... (또는 "col1", "col2", ...)로 재네이밍합니다.
    
    Args:
        rows (dict): {'equations': {label: equation, ...}, 'points': {label: point, ...}}
                      각 equation은 [a, b, c, xmin, xmax, ...] 형식의 리스트.
        cols (dict): {'equations': {label: equation, ...}, 'points': {label: point, ...}}
                      각 equation은 [a, b, c, ymin, ymax, ...] 형식의 리스트.
                      
    Returns:
        tuple: (rows, cols) - 업데이트된 딕셔너리들.
    """
    
    def compute_max_curvature(equation, dom_min, dom_max):
        """
        주어진 2차 방정식 (리스트 형식: [a, b, c, ...])에 대해,
        독립변수 t에 대한 최대 곡률을 구합니다.
        여기서 t는 x 또는 y가 될 수 있으며, 허용 구간은 [dom_min, dom_max].
        """
        a, b = equation[0], equation[1]
        # a가 0이면 직선이므로 곡률은 0
        if a == 0:
            return 0.0
        # 정점 위치 t_v = -b/(2a)
        t_v = -b / (2 * a)
        if dom_min <= t_v <= dom_max:
            return abs(2 * a)
        else:
            # 양쪽 경계에서의 곡률 계산
            k1 = abs(2 * a) / ((1 + (2 * a * dom_min + b) ** 2) ** (3 / 2))
            k2 = abs(2 * a) / ((1 + (2 * a * dom_max + b) ** 2) ** (3 / 2))
            return max(k1, k2)

    def filter_labels(data, var_type='x', prefix='row'):
        """
        data 딕셔너리에서 'equations'에 대해 원래 레이블 순서를 유지하면서
        각 방정식의 최대 곡률을 계산하고, threshold 미만인 경우 제거합니다.
        이후 남은 항목들을 prefix와 순번으로 재네이밍합니다.
        
        var_type: 'x' 또는 'y' - 해당 독립변수에 대한 허용 구간 사용.
        """
        original_keys = list(data['equations'].keys())
        kept_keys = []
        for key in original_keys:
            eq = data['equations'][key]
            # rows: eq[3]=xmin, eq[4]=xmax; cols: eq[3]=ymin, eq[4]=ymax
            dom_min, dom_max = eq[3], eq[4]
            kappa_max = compute_max_curvature(eq, dom_min, dom_max)
            if kappa_max <= threshold:
                kept_keys.append(key)
        # 재네이밍: kept_keys 순서대로 새 딕셔너리 생성
        new_equations = {}
        new_points = {}
        for idx, old_key in enumerate(kept_keys, start=1):
            new_key = f"{prefix}{idx}"
            new_equations[new_key] = data['equations'][old_key]
            new_points[new_key] = data['points'][old_key]
        data['equations'] = new_equations
        data['points'] = new_points
        return data

    # rows: 독립변수가 x이므로 prefix "row"
    rows = filter_labels(rows, var_type='x', prefix='row')
    # cols: 독립변수가 y이므로 prefix "col"
    cols = filter_labels(cols, var_type='y', prefix='col')
    
    return rows, cols


def indexing_data(rows, cols, input_image, mask_contour, circle_radius0, epsilon=1e-6):
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
        # if circle_radius0 <= 5:
        #     circle_radius0 = circle_radius0 + 10
        half_size = int(circle_radius0 / 5)  # 10x10 영역 (중심 기준 ±10)
        half_size = max(half_size, 3)
        if half_size > 10: half_size = half_size + 5
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
            point_id = (new_col_index, new_row_index)
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
            point_id = (new_col_index, new_row_index)
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
                    "id": (new_col_index, new_row_index),
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



def draw_points(original_img, rows_removed, cols_removed, center_point, highlight=False, cols_highlight=None):
    """
    Draws points from rows_removed and cols_removed on separate images with unique colors per label.
    highlight가 True이면, cols_removed의 모든 점은 보라색으로 그리고, cols_highlight에 해당하는 라벨은 초록색으로 표시합니다.

    Args:
        original_img (numpy.ndarray): The original image in BGR format.
        rows_removed (dict): Dictionary with row labels as keys and lists of points as values.
        cols_removed (dict): Dictionary with column labels as keys and lists of points as values.
        center_point (tuple): 중심 좌표.
        highlight (bool): True이면 cols_removed 점들을 특별히 표시합니다.
        cols_highlight (list): highlight 시 초록색으로 표시할 라벨 리스트.

    Returns:
        tuple: (row_img, col_img) with points drawn.
    """
    # Create copies of the original image for rows and columns
    row_img = original_img.copy()
    col_img = original_img.copy()

    # Generate distinct colors for rows and columns (when highlight is False)
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

    # 기존 컬럼 색상은 highlight가 False일 경우에만 사용
    col_labels = sorted(cols_removed.keys())
    num_cols = len(col_labels)
    col_colors = generate_colors(num_cols)
    col_color_dict = {label: color for label, color in zip(col_labels, col_colors)}

    # Define drawing parameters
    point_radius = 3
    point_thickness = -1  # Filled circle

    # Draw rows_removed points on row_img
    for label, points in rows_removed.items():
        color = row_color_dict.get(label, (0, 255, 0))  # 기본은 초록색
        for point in points:
            x = int(point['x'])
            y = int(point['y'])
            cv2.circle(row_img, (x, y), point_radius, color, point_thickness)

    # Draw cols_removed points on col_img
    if highlight:
        # highlight=True인 경우, 모든 점은 보라색으로 그리고, 특정 라벨은 초록색으로 표시
        # 보라색: BGR에서 (255, 0, 255), 초록색: (0, 255, 0)
        if cols_highlight is None:
            cols_highlight = []
        for label, points in cols_removed.items():
            # highlight 대상이면 초록색, 아니면 보라색
            color = (0, 255, 0) if label in cols_highlight else (255, 0, 255)
            for point in points:
                x = int(point['x'])
                y = int(point['y'])
                cv2.circle(col_img, (x, y), point_radius, color, point_thickness)
    else:
        # highlight가 False인 경우 기존 로직 사용 (각 라벨별 고유 색상)
        for label, points in cols_removed.items():
            color = col_color_dict.get(label, (255, 0, 0))  # 기본은 파란색
            for point in points:
                x = int(point['x'])
                y = int(point['y'])
                cv2.circle(col_img, (x, y), point_radius, color, point_thickness)

    # 중심 좌표에 원 그리기 (예: 파란색 원)
    cv2.circle(col_img, (int(center_point[0]), int(center_point[1])), 5, (255, 0, 0), 1)

    return row_img, col_img

def remove_minus_labels(cols_dict):
    """
    cols_dict: 예를 들어
      {'col-11': [...], 'col-10': [...], 'col-9': [...], 'col0': [...], ...}
    
    "col-"로 시작하는 키를 제거하고, 나머지 키들만 포함하는
    새로운 딕셔너리를 반환합니다.
    """
    new_cols = {}
    for key, value in cols_dict.items():
        if not key.startswith("col-"):
            new_cols[key] = value
    return new_cols




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

# @profile
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


def mask_roi_around_center(horizontal_mask, vertical_mask, mask_contour, original_img):
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
    
    # 2. GaussianBlur 적용 (커널 크기 19x19)
    blurred = cv2.GaussianBlur(image_gray, (19, 19), 0)
    
    # 3. 밝기 기준 240로 threshold 적용하여 이진화
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
        circle_radius0 = int(circle_radius)
        
        # circle_radius에 따라 보정값 적용 (예: 반지름이 작으면 +20, 그렇지 않으면 +5)
        if circle_radius < 30:
            circle_radius = circle_radius0 + 20
        else:
            circle_radius = circle_radius0 + 5
        
        # 타원 그리기: 단축(작은 축) = circle_radius - 20, 장축(큰 축) = circle_radius + 20
        # cv2.ellipse의 axes 인수는 반축 길이이므로, 각각 0.5*(circle_radius-20)와 0.5*(circle_radius+20)
        minor_axis = max(circle_radius+20, 1)  # 음수가 되지 않도록 보정
        axes = (int(round((circle_radius + 40) / 2)), int(round(minor_axis / 2)))
        cv2.ellipse(circle_mask, circle_center, axes, 0, 0, 360, 0, thickness=-1)
    
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
    
    return mask_roi_h, mask_roi_v, circle_radius0


# kernprof -v -l python_grid_detection_cylinder.py
# python -m line_profiler python_grid_detection_cylinder.py.lprof > results_main.txt

# @profile
def color_and_expand_lines(mask_roi_h, mask_roi_v, circle_radius0, center_point, max_contour, mask_contour, original_img, cylinder_centroids):
    """
    마스크를 라벨링하여 색상화, 라인 확장(arc fitting)까지 수행
    """
    if max_contour is not None:
        x, y, w, h = cv2.boundingRect(max_contour)

        # # rows, cols 끊어진 roi 확장해서 잇기
        horizontal_expanded, horizontal_dubug = expands_line_roi(mask_roi_h, 1, mask_contour, kernel_size=(91 + circle_radius0))
        vertical_expanded, vertical_dubug = expands_line_roi(mask_roi_v, 1, mask_contour, kernel_size=(91 + circle_radius0))

        # rows, cols label 설정(row1, row2, ,,,) 준비
        distinct_colors = generate_distinct_colors(20)
        num_labels_h2, labels_h2, horizontal_colored2 = label_and_color_masks(horizontal_expanded[y:y+h, x:x+w],generate_unique_colors(20, distinct_colors))
        num_labels_v2, labels_v2, vertical_colored2 = label_and_color_masks(vertical_expanded[y:y+h, x:x+w],generate_unique_colors(20, distinct_colors))

        # label 설정
        rows = group_points_by_label(cylinder_centroids, labels_h2, x, y)
        cols = group_points_by_label(cylinder_centroids, labels_v2, x, y)

        # rows, cols ['equations'] 초기화 (degree=2)
        rows, cols = create_dummy_rows_cols(rows, cols, degree=2)

        # rows, cols ['points'] 에 degree=3인 poly fitting => 비정상 label 제거 => fitted poly간 intesrection point 구하기
        img_with_poly, rows, cols = fit_and_draw_polynomial(original_img, rows, cols, w, h, max_contour, degree=2)        
        rows, cols = remove_label(rows, cols)
        # img_with_poly_G, rows_G, cols_G = modify_grayscale_Cline(original_img, rows, cols, draw_points=True, degree=degree, window_size=7)             
        img_with_points, rows_updated, cols_updated = find_and_assign_intersections_P(img_with_poly, rows, cols, max_contour, draw_points=True, degree=2)

        # 비어있는 label 제거 및 label이 1부터 시작하게끔 re-naming
        rows_updated, cols_updated = clean_and_relabel(rows_updated, cols_updated)
        
        # center point 구하고 center에 맞춰서 row, col indexing
        result_json, result_dict, rows_dict, cols_dict, center_point = indexing_data(
            rows_updated, cols_updated, original_img, mask_contour, circle_radius0
        )

        # matlab으로 넘기기 위해 json 변환
        cols_dict = remove_minus_labels(cols_dict)
        row_img, col_img = draw_points(original_img, rows_dict, cols_dict, center_point, highlight=False)

        result_json = make_json(center_point, cols_dict)
        
        return col_img, result_json, rows_updated, cols_updated

    # max_contour가 None인 경우, 그냥 결과 없는 값
    return original_img, json.dumps({}), {}, {}


