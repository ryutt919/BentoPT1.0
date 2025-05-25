# pose_estimator.py
# MediaPipe를 이용해 관절 위치와 주요 중심점을 추정하는 모듈

import cv2 as cv
import mediapipe as mp
from config import JOINT_INDICES

# MediaPipe Pose 초기화
mp_pose = mp.solutions.pose
_pose = mp_pose.Pose(static_image_mode=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

def get_pose_landmarks(frame):
    """
    2D, 3D 랜드마크를 함께 반환
    - landmarks_2d: {index: (x, y)}
    - landmarks_3d: {index: (x, y, z)}
    """
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = _pose.process(rgb)
    landmarks_2d, landmarks_3d = {}, {}
    if results.pose_landmarks:
        h, w = frame.shape[:2]
        for i, lm in enumerate(results.pose_landmarks.landmark):
            landmarks_2d[i] = (int(lm.x * w), int(lm.y * h))
            landmarks_3d[i] = (lm.x, lm.y, lm.z)
    return landmarks_2d, landmarks_3d


# --- 수정된 부분 시작 ---
def get_shoulder_center(landmarks):
    """왼쪽(11)과 오른쪽(12) 어깨의 중간 좌표 계산."""
    if 11 in landmarks and 12 in landmarks:
        x = (landmarks[11][0] + landmarks[12][0]) // 2
        y = (landmarks[11][1] + landmarks[12][1]) // 2
        return (x, y)
    return None

def get_hip_center(landmarks):
    """왼쪽(23)과 오른쪽(24) 골반의 중간 좌표 계산."""
    if 23 in landmarks and 24 in landmarks:
        x = (landmarks[23][0] + landmarks[24][0]) // 2
        y = (landmarks[23][1] + landmarks[24][1]) // 2
        return (x, y)
    return None

def estimate_spine(landmarks):
    """
    어깨 중심과 골반 중심의 중간점을 척추 위치로 추정.
    """
    shoulder = get_shoulder_center(landmarks)
    hip      = get_hip_center(landmarks)
    if shoulder and hip:
        x = (shoulder[0] + hip[0]) // 2
        y = hip[1]
        return (x, y)
    return None

def estimate_head_center_priority(landmarks):
    """
    머리 중심 추정 우선순위:
      1) 코(0) → 2) 귀(7,8 평균) → 3) 없으면 None
    """
    # 1) 코
    if 0 in landmarks:
        return landmarks[0]
    # 2) 귀
    if 7 in landmarks and 8 in landmarks:
        x = (landmarks[7][0] + landmarks[8][0]) // 2
        y = (landmarks[7][1] + landmarks[8][1]) // 2
        return (x, y)
    # 3) 둘 다 없으면
    return None
# --- 수정된 부분 끝 ---
