# main.py

import time
import cv2 as cv
import tkinter as tk
from tkinter import filedialog

from config import (
    PAUSE_KEY, FLIP_KEY, RECORD_KEY,
    SCALE, THICKNESS,
    CONNECTIONS_BASIC, COLORS,
    EXERCISE_LIST, CONNECTIONS_LEG
)
from pose_estimator import (
    get_pose_landmarks,
    get_head_center, get_shoulder_center,
    get_hip_center, estimate_spine
)
from exercise import ExerciseBase
from visualizer import (
    draw_landmarks,
    draw_connections,
    draw_feedback,
    draw_centers
)
WINDOW_NAME = 'Pose Tracker'


# 전역 상태 변수
paused    = False
flipped   = False
recording = False
writer    = None

def select_source():
    """시작 시 GUI로 파일 또는 웹캠 선택."""
    root = tk.Tk()
    root.title("소스 선택")
    root.geometry("300x100")
    choice = {'source': 0}

    def choose_file():
        path = filedialog.askopenfilename(
            title="동영상 파일 선택",
            filetypes=[("Video Files", "*.mp4;*.avi;*.mov"), ("All Files", "*.*")]
        )
        if path:
            choice['source'] = path
        root.quit()

    def use_camera():
        choice['source'] = 0
        root.quit()

    tk.Button(root, text="파일 선택", width=20, command=choose_file).pack(pady=5)
    tk.Button(root, text="웹캠 사용", width=20, command=use_camera).pack(pady=5)

    root.mainloop()
    root.destroy()
    return choice['source']

def select_exercise():
    """GUI로 운동 종목 선택."""
    root = tk.Tk()
    root.title("운동 선택")
    root.geometry("300x120")
    choice = {'exercise': EXERCISE_LIST[0]}

    var = tk.StringVar(root)
    var.set(EXERCISE_LIST[0])
    tk.Label(root, text="운동을 선택하세요:").pack(pady=(10,0))
    tk.OptionMenu(root, var, *EXERCISE_LIST).pack(pady=5)

    def confirm():
        choice['exercise'] = var.get()
        root.quit()

    tk.Button(root, text="확인", width=10, command=confirm).pack(pady=10)

    root.mainloop()
    root.destroy()
    return choice['exercise']

def handle_keys(key, frame_size):
    """Space/F/R 키로 pause, flip, record 토글."""
    global paused, flipped, recording, writer

    w, h = frame_size

    if key == ord(PAUSE_KEY):
        paused = not paused

    elif key == ord(FLIP_KEY):
        flipped = not flipped

    elif key == ord(RECORD_KEY):
        recording = not recording
        if recording:
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            writer = cv.VideoWriter('output.mp4', fourcc, 30.0, (w, h))
        else:
            if writer is not None:
                writer.release()
                writer = None

def main():
    # 1) 비디오 소스 선택
    source = select_source()
    cap = cv.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: cannot open source {source}")
        return

    # 2) 운동 선택
    exercise_name = select_exercise()
    exercise = ExerciseBase(exercise_name)

    # 3) 프레임 크기
    width  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    # 4) 메인 루프
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if flipped:
            frame = cv.flip(frame, 1)

        landmarks = get_pose_landmarks(frame)
        now       = time.time()

        # 운동 분석
        count, feedback_list = exercise.update(landmarks, now)

        # 중심점 계산
        centers = {
            'head':     get_head_center(landmarks),
            'shoulder': get_shoulder_center(landmarks),
            'hip':      get_hip_center(landmarks),
            'spine':    estimate_spine(landmarks)
        }

        # 시각화
        draw_landmarks(frame, landmarks, SCALE, THICKNESS)
        draw_connections(frame, landmarks, CONNECTIONS_BASIC, THICKNESS)
        draw_centers(frame, centers, THICKNESS)
        Visualizer.draw_connections(frame, landmarks, CONNECTIONS_LEG, THICKNESS)
        # exercise.update가 반환한 feedback_list를 바로 렌더링
        draw_feedback(frame, feedback_list, SCALE, THICKNESS)

        # 녹화
        if recording and writer:
            writer.write(frame)

        # 화면 출력
        cv.imshow('Pose Tracker', frame)

        key = cv.waitKey(1) & 0xFF
        handle_keys(key, (width, height))
              
        if key == 27:  # ESC
            break
        if paused:
            handle_keys(cv.waitKey(), (width, height))
           
    cap.release()
    if writer:
        writer.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
