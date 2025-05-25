# utils.py
import numpy as np
import cv2 as cv
import math

def calculate_angle(a, b, c):
    a, b, c = map(np.array, (a, b, c))
    ab = a - b; cb = c - b
    cosv = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
    cosv = np.clip(cosv, -1.0, 1.0)
    return np.degrees(np.arccos(cosv))

def get_camera_orientation(landmarks_2d, frame_shape,
                           model_shoulder_width=0.4, model_torso_height=0.5):
    h, w = frame_shape[:2]
    # 카메라 내부 파라미터
    focal = w
    camera_matrix = np.array([
        [focal,   0,    w/2],
        [  0,   focal,  h/2],
        [  0,     0,     1 ]
    ], float)
    dist_coeffs = np.zeros((4,1))
    # 3D 모델 포인트 (어깨2, 골반2)
    model_points = np.array([
        (-model_shoulder_width/2, 0.0, 0.0),
        ( model_shoulder_width/2, 0.0, 0.0),
        (-model_shoulder_width/2, -model_torso_height, 0.0),
        ( model_shoulder_width/2, -model_torso_height, 0.0),
    ], float)
    image_points = np.array([
        landmarks_2d[11], landmarks_2d[12],
        landmarks_2d[23], landmarks_2d[24],
    ], float)
    ok, rvec, tvec = cv.solvePnP(
        model_points, image_points,
        camera_matrix, dist_coeffs,
        flags=cv.SOLVEPNP_ITERATIVE
    )
    R, _ = cv.Rodrigues(rvec)
    sy = math.hypot(R[0,0], R[1,0])
    yaw  = math.degrees(math.atan2(-R[2,0], sy))
    roll = math.degrees(math.atan2(R[1,0],  R[0,0]))
    return yaw, roll
