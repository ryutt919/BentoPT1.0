# main.py
import cv2 as cv
import time
from config import JOINT_INDICES, CONNECTIONS_BASIC
from pose_estimator import (
    get_pose_landmarks,
    estimate_spine,
    estimate_head_center_priority,
    get_shoulder_center
)
from visualizer import (
    draw_joints,
    draw_connections,
    draw_limbs,
    draw_torso,
    draw_shoulder_connection
)``
from analyzer import evaluate_pushup
from feedback import draw_feedback
from utils import get_camera_orientation

VIDEO_PATH    = './testVD.mp4'
scale         = 1.0
YAW_THRESHOLD = 10.0
ROLL_THRESHOLD= 10.0

def main():
    cap = cv.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("영상을 열 수 없습니다."); return

    fps   = cap.get(cv.CAP_PROP_FPS)
    delay = int(1000/fps) if fps>0 else 33
    last_print = time.time()

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1) 2D/3D 랜드마크
        lm2d, lm3d = get_pose_landmarks(frame)

        # 2) 시각화 (기본선: 어깨 제외)
        draw_joints(frame, lm2d, JOINT_INDICES)
        filtered = [p for p in CONNECTIONS_BASIC if set(p)!={11,12}]
        draw_connections(frame, lm2d, filtered)
        draw_limbs(frame, lm2d, 'left')
        draw_limbs(frame, lm2d, 'right')

        # 3) 몸통·머리
        sc = get_shoulder_center(lm2d)
        sp = estimate_spine(lm2d)
        hc = estimate_head_center_priority(lm2d)
        draw_torso(frame, sc, sp, hc)

        # 4) 푸시업 평가
        msg = evaluate_pushup(lm2d)
        draw_feedback(frame, msg)

        # 5) 카메라 정면 여부 (PnP → yaw/roll)
        if all(k in lm2d for k in (11,12,23,24)):
            yaw, roll = get_camera_orientation(lm2d, frame.shape)
            now = time.time()
            if now - last_print >= 1.0:
                print(f"[Camera yaw] {yaw:.1f}°, [roll] {roll:.1f}°")
                last_print = now
            is_frontal = abs(yaw)<YAW_THRESHOLD and abs(roll)<ROLL_THRESHOLD
            draw_shoulder_connection(frame, lm2d, is_frontal)

        # 6) 최종 출력
        frame = cv.resize(frame, None, fx=scale, fy=scale)
        # frame = cv.flip(frame, 1)
        cv.imshow('Pose Trainer', frame)
        if cv.waitKey(delay) & 0xFF == 27:
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
