import cv2 as cv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import numpy as np
from scipy.ndimage import median_filter, gaussian_filter
from collections import deque
from BTPT_pose import PoseEstimator
from BTPT_vis import PoseVisualizer
import json
from pathlib import Path



class ExerciseFeedbackApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("BTPT 1.0")
        self.root.geometry("400x400")  # 창 크기 증가
        
        # 기본 폰트 설정
        default_font = ('Malgun Gothic', 10)  # 한글 지원 폰트
        self.root.option_add("*Font", default_font)
        
        # 키포인트 버퍼 초기화 (시간 차원 처리용)
        self.keypoints_buffer = deque(maxlen=32)  # ST-GCN 입력용 32 프레임 버퍼
        
        # 창을 화면 중앙에 위치
        window_width = 400
        window_height = 400
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        center_x = int(screen_width/2 - window_width/2)
        center_y = int(screen_height/2 - window_height/2)
        self.root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        
        # 운동 종류 선택
        self.exercise_frame = ttk.LabelFrame(self.root, text="운동 종류 선택", padding="10")
        self.exercise_frame.pack(fill="x", padx=10, pady=5)
        
        # 운동 종류와 한글 이름 매핑
        self.exercise_mapping = {
            "런지": "lunge",
            "풀업": "pull_up"
        }
        
        self.exercise_types = ["런지", "풀업"]
        self.selected_exercise = tk.StringVar()
        self.selected_exercise.set(self.exercise_types[0])
        
        for exercise in self.exercise_types:
            ttk.Radiobutton(
                self.exercise_frame, 
                text=exercise, 
                variable=self.selected_exercise, 
                value=exercise
            ).pack(anchor="w")

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
        
        # 라벨 맵 로드
        self.label_map = self._load_label_map()
        
        # 결과 표시를 위한 레이블 추가 (폰트 크기 증가)
        self.result_label = tk.Label(
            self.root, 
            text="분석 결과", 
            font=('Malgun Gothic', 14, 'bold')
        )
        self.result_label.pack(pady=10)
        
        # 상세 결과를 표시할 텍스트 위젯 (폰트 설정 추가)
        self.result_text = tk.Text(
            self.root, 
            height=10,  # 높이 증가
            width=60,   # 너비 증가
            font=('Malgun Gothic', 12),
            wrap=tk.WORD  # 자동 줄바꿈
        )
        self.result_text.pack(pady=5, padx=10)

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
        selected_exercise = self.exercise_mapping[self.selected_exercise.get()]
        video_path = self.video_path.get()
        
        if not video_path:
            tk.messagebox.showerror("오류", "영상을 선택해주세요.")
            return
            
        # 포즈 추출기와 시각화 도구 초기화 - exercise_type 전달
        pose_estimator = PoseEstimator(exercise_type=selected_exercise)
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
                frame = visualizer.draw_pose(frame, keypoints)
                result = pose_estimator.process_frame(frame, frame_count)
                
                # GUI 업데이트
                if result is not None:
                    frame = visualizer.draw_pose(frame, keypoints, result)
                    self.update_result_display(result)
            
            frame_count += 1
        
            # 화면에 맞게 크기 조정 (가로/세로 중 큰 쪽을 640으로)
            height, width = frame.shape[:2]
            target_size = 640
            if width > height:
                new_width = target_size
                new_height = int(height * (target_size / width))
            else:
                new_height = target_size
                new_width = int(width * (target_size / height))

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

    def _load_label_map(self):
        """라벨 맵 JSON 파일 로드"""
        label_map = {
            0: "판별 불가",
            1: "올바른 자세",
            2: "무릎 90도 미만 주의의",
            3: "등이 구부러짐",
            4: "발이 불안정함",
            5: "몸통이 흔들림",
            6: "뒷무릎이 바닥에 닿음",
            7: "최대 수축 필요",
            8: "최대 이완 필요"
        }
        return label_map
            
    def update_result_display(self, result):
        """분석 결과를 GUI에 표시 - 피드백만 표시"""
        self.result_text.delete('1.0', tk.END)
        
        if result is None or not result.get("feedback"):
            self.result_text.insert(tk.END, "피드백이 없습니다.")
            return
            
        # 피드백만 표시
        for feedback in result["feedback"]:
            self.result_text.insert(tk.END, f"• {feedback['message']}\n")

if __name__ == "__main__":
    app = ExerciseFeedbackApp()
    app.root.mainloop()
