# pose_estimator.py
# MediaPipe를 이용해 관절 위치와 주요 중심점을 추정하는 모듈

import cv2 as cv
import mediapipe as mp
from config import JOINT_INDICES, HEAD_LANDMARKS

# MediaPipe Pose 초기화
mp_pose = mp.solutions.pose
_pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def get_pose_landmarks(frame):
    """
    주어진 프레임에서 MediaPipe Pose를 사용해
    JOINT_INDICES 및 HEAD_LANDMARKS에 정의된 랜드마크들의
    2D 픽셀 좌표를 반환합니다.

    Returns:
        landmarks (dict): {landmark_index: (x, y)}
    """
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = _pose.process(rgb)
    h, w = frame.shape[:2]
    landmarks = {}
    if results.pose_landmarks:
        for idx in set(JOINT_INDICES) | set(HEAD_LANDMARKS.values()):
            lm = results.pose_landmarks.landmark[idx]
            if hasattr(lm, 'visibility') and lm.visibility < 0.5:
                continue
            x, y = int(lm.x * w), int(lm.y * h)
            landmarks[idx] = (x, y)
    return landmarks


def get_head_center(landmarks):
    """
    머리 중심 좌표를 추정합니다:
      1) 양쪽 귀(LEFT, RIGHT) 모두 검출 시, 두 귀의 중간점
      2) 그 외 한쪽 귀만이거나 둘 다 없으면, 코(NOSE)
      3) 코도 없으면 None

    Args:
        landmarks (dict): get_pose_landmarks의 반환값
    Returns:
        (x, y) 또는 None
    """
    left_idx = HEAD_LANDMARKS['left_ear']
    right_idx = HEAD_LANDMARKS['right_ear']
    nose_idx = HEAD_LANDMARKS['nose']
    if left_idx in landmarks and right_idx in landmarks:
        lx, ly = landmarks[left_idx]
        rx, ry = landmarks[right_idx]
        return ((lx + rx) // 2, (ly + ry) // 2)
    if nose_idx in landmarks:
        return landmarks[nose_idx]
    return None


def get_shoulder_center(landmarks):
    """왼쪽(11)과 오른쪽(12) 어깨의 중간 좌표 계산."""
    left_sh = 11
    right_sh = 12
    if left_sh in landmarks and right_sh in landmarks:
        x = (landmarks[left_sh][0] + landmarks[right_sh][0]) // 2
        y = (landmarks[left_sh][1] + landmarks[right_sh][1]) // 2
        return (x, y)
    return None


def get_hip_center(landmarks):
    """왼쪽(23)과 오른쪽(24) 골반의 중간 좌표 계산."""
    left_hip = 23
    right_hip = 24
    if left_hip in landmarks and right_hip in landmarks:
        x = (landmarks[left_hip][0] + landmarks[right_hip][0]) // 2
        y = (landmarks[left_hip][1] + landmarks[right_hip][1]) // 2
        return (x, y)
    return None


def estimate_spine(landmarks):
    """
    어깨 중심과 골반 중심의 중간점을 척추 위치로 추정.
    Returns: (x, y) 또는 None
    """
    shoulder = get_shoulder_center(landmarks)
    hip      = get_hip_center(landmarks)
    if shoulder and hip:
        x = (shoulder[0] + hip[0]) // 2
        y = hip[1]
        return (x, y)
    return None
