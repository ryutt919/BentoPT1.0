# utils.py
# 공통 연산 유틸리티: 벡터/각도 계산, 스무딩, 값 보정 등

import numpy as np


def get_vector(p1, p2):
    """
    p1에서 p2로 향하는 2D 벡터를 반환합니다.

    Args:
        p1 (tuple): (x, y)
        p2 (tuple): (x, y)
    Returns:
        np.ndarray: [dx, dy]
    """
    return np.array([p2[0] - p1[0], p2[1] - p1[1]], dtype=float)


def get_angle(p1, p2, p3):
    """
    p2를 꼭짓점으로 하는 p1-p2-p3 각도를 도 단위로 계산합니다.

    Args:
        p1, p2, p3 (tuple): (x, y)
    Returns:
        float: 두 벡터 사이 각도 (0~180)
    """
    v1 = get_vector(p2, p1)
    v2 = get_vector(p2, p3)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    cos_val = np.dot(v1, v2) / (norm1 * norm2)
    cos_val = np.clip(cos_val, -1.0, 1.0)
    angle_rad = np.arccos(cos_val)
    return np.degrees(angle_rad)


def moving_average(data, window_size):
    """
    이동 평균 스무딩을 적용합니다.

    Args:
        data (list): 숫자 또는 (x, y) 튜플 리스트
        window_size (int): 윈도우 크기
    Returns:
        list: 스무딩된 데이터
    """
    if len(data) < window_size:
        return data.copy()
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window_size + 1)
        window = data[start:i+1]
        arr = np.array(window, dtype=float)
        avg = np.mean(arr, axis=0) if arr.ndim > 1 else np.mean(arr)
        # 튜플 형태로 반환
        smoothed.append(tuple(avg) if arr.ndim > 1 else float(avg))
    return smoothed


def clamp(val, min_val, max_val):
    """
    값을 [min_val, max_val] 범위로 제한합니다.
    """
    return max(min_val, min(val, max_val))


def format_time(delta):
    """
    시간 차이를 '0.00s' 형식 문자열로 반환합니다.
    """
    return f"{delta:.2f}s"
