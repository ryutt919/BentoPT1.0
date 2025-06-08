import cv2 as cv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import numpy as np
from scipy.ndimage import median_filter, gaussian_filter
from collections import deque
from BTPT_pose import PoseEstimator
from BTPT_vis import PoseVisualizer

class KalmanFilter:
    def __init__(self, process_variance=1e-4, measurement_variance=1e-2):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.posteri_estimate = 0.0
        self.posteri_error_estimate = 1.0
        
    def update(self, measurement):
        priori_estimate = self.posteri_estimate
        priori_error_estimate = self.posteri_error_estimate + self.process_variance
        
        blending_factor = priori_error_estimate / (priori_error_estimate + self.measurement_variance)
        self.posteri_estimate = priori_estimate + blending_factor * (measurement - priori_estimate)
        self.posteri_error_estimate = (1 - blending_factor) * priori_error_estimate
        
        return self.posteri_estimate

class ExerciseFeedbackApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("운동 피드백 시스템")
        self.root.geometry("400x300")
          # 키포인트 버퍼 초기화 (시간 차원 처리용)
        self.keypoints_buffer = deque(maxlen=32)  # ST-GCN 입력용 32 프레임 버퍼
        
        # 창을 화면 중앙에 위치
        window_width = 400
        window_height = 300
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        center_x = int(screen_width/2 - window_width/2)
        center_y = int(screen_height/2 - window_height/2)
        self.root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        
        # 운동 종류 선택
        self.exercise_frame = ttk.LabelFrame(self.root, text="운동 종류 선택", padding="10")
        self.exercise_frame.pack(fill="x", padx=10, pady=5)
        
        self.exercise_types = ["런지", "풀업"]
        self.selected_exercise = tk.StringVar()
        self.selected_exercise.set(self.exercise_types[0])
        
        for exercise in self.exercise_types:
            ttk.Radiobutton(self.exercise_frame, text=exercise, 
                          variable=self.selected_exercise, value=exercise).pack(anchor="w")
        
        # 영상 선택
        self.video_frame = ttk.LabelFrame(self.root, text="영상 선택", padding="10")
        self.video_frame.pack(fill="x", padx=10, pady=5)
        
        self.video_path = tk.StringVar()
        self.video_label = ttk.Label(self.video_frame, text="선택된 영상: 없음")
        self.video_label.pack(fill="x")
        
        self.select_button = ttk.Button(self.video_frame, text="영상 파일 선택", 
                                      command=self.select_video)
        self.select_button.pack(pady=5)
        
        # 시작 버튼
        self.start_button = ttk.Button(self.root, text="분석 시작", 
                                     command=self.start_analysis)
        self.start_button.pack(pady=20)
        
        # 칼만 필터 초기화 (각 키포인트의 x, y 좌표에 대해)
        self.kalman_filters = {}

    def select_video(self):
        filetypes = (
            ('Video files', '*.mp4 *.avi *.mov'),
            ('All files', '*.*')
        )
        
        filename = filedialog.askopenfilename(
            title='영상 파일을 선택하세요',
            filetypes=filetypes,
            initialdir='./videos'
        )
        
        if filename:
            self.video_path.set(filename)
            self.video_label.config(text=f"선택된 영상: {os.path.basename(filename)}")

    def start_analysis(self):
        selected_exercise = self.selected_exercise.get()
        video_path = self.video_path.get()
        
        if not video_path:
            tk.messagebox.showerror("오류", "영상을 선택해주세요.")
            return
            
        # 포즈 추출기와 시각화 도구 초기화
        pose_estimator = PoseEstimator()
        visualizer = PoseVisualizer()
        
        # 비디오 캡처 객체 생성
        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            tk.messagebox.showerror("오류", "영상을 열 수 없습니다.")
            return
            
        # 프레임 카운터
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 포즈 추출 및 분석
            keypoints = pose_estimator.extract_keypoints(frame, frame_count)
            
            if keypoints is not None:
                # 키포인트 오버레이 추가
                frame = visualizer.draw_pose(frame, keypoints)
                
                # 분석 결과 처리
                result = pose_estimator.process_frame(frame, frame_count)
                if result is not None:
                    frame = visualizer.draw_pose(frame, keypoints, result)
                frame_count += 1
            
            # 화면에 맞게 크기 조정
            height, width = frame.shape[:2]
            max_dim = 800
            if height > max_dim or width > max_dim:
                scale = max_dim / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv.resize(frame, (new_width, new_height))
            
            # 프레임 표시
            cv.imshow(f"Exercise Analysis - {selected_exercise}", frame)
            
            # q를 누르면 종료
            key = cv.waitKey(1)
            if key == ord('q'):
                break
                
            frame_count += 1
        
        # 정리
        cap.release()
        cv.destroyAllWindows()
        print(f"분석 완료: {selected_exercise}")
    def smooth_keypoints(self, keypoints):
        """키포인트를 필터링 없이 그대로 반환합니다."""
        if keypoints is None:
            return None
        return keypoints

if __name__ == "__main__":
    app = ExerciseFeedbackApp()
    app.root.mainloop()