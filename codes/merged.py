import time
import cv2 as cv
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageDraw, ImageFont
import mediapipe as mp
from config import (
    PAUSE_KEY, FLIP_KEY, RECORD_KEY,
    SCALE, THICKNESS,
    CONNECTIONS_BASIC, COLORS,
    EXERCISE_LIST, HEAD_LANDMARKS, JOINT_INDICES,
    FONT_PATHS, FEEDBACK_MESSAGES, EXERCISE_PARAMS
)

# ========== UTILS ==========
class Utils:
    @staticmethod
    def get_vector(p1, p2):
        return np.array([p2[0] - p1[0], p2[1] - p1[1]], dtype=float)
    @staticmethod
    def get_angle(p1, p2, p3):
        v1 = Utils.get_vector(p2, p1)
        v2 = Utils.get_vector(p2, p3)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        cos_val = np.dot(v1, v2) / (norm1 * norm2)
        cos_val = np.clip(cos_val, -1.0, 1.0)
        angle_rad = np.arccos(cos_val)
        return np.degrees(angle_rad)
    @staticmethod
    def angle_between_lines(a1, a2, b1, b2):
        # 두 선분의 벡터 각도 (0~180)
        v1 = Utils.get_vector(a1, a2)
        v2 = Utils.get_vector(b1, b2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        cos_val = np.dot(v1, v2) / (norm1 * norm2)
        cos_val = np.clip(cos_val, -1.0, 1.0)
        return np.degrees(np.arccos(cos_val))
    @staticmethod
    def angle_with_vertical(p1, p2):
        # 선분(p1-p2)이 화면 y축(0,1)과 이루는 각도(절대값)
        v = Utils.get_vector(p1, p2)
        vertical = np.array([0, 1])
        norm_v = np.linalg.norm(v)
        if norm_v == 0:
            return 0.0
        cos_val = np.dot(v, vertical) / norm_v
        cos_val = np.clip(cos_val, -1.0, 1.0)
        return abs(np.degrees(np.arccos(cos_val)))
    @staticmethod
    def format_time(delta):
        return f"{delta:.2f}s"

# ========== REP COUNTER: PULLUP ANGLE ONLY ==========
class PullupCounter:
    def __init__(self):
        self.state = 'init'
        self.count = 0
        self.last_transition_time = None
        self.min_angle = 999
        self.min_angle_landmarks = None
        self.last_valid_min_feedback = None
        self.last_angle3_value = None  # <--- 추가(각도 유지)
    def update(self, landmarks, now=None):
        try:
            shoulder = landmarks[11]
            elbow    = landmarks[13]
            wrist    = landmarks[15]
        except KeyError:
            return 0, None
        angle = Utils.get_angle(shoulder, elbow, wrist)
        if angle < self.min_angle:
            self.min_angle = angle
            self.min_angle_landmarks = {k: landmarks.get(k) for k in (11, 12, 13, 15, 16)}
        if now is None:
            now = time.time()
        count_inc = 0
        if self.state == 'init':
            if angle > 150:
                self.state = 'down'
            elif angle < 70:
                self.state = 'up'
            self.last_transition_time = now
            return 0, None
        if self.state == 'down':
            if angle < 70:
                self.state = 'up'
                # self.last_valid_min_feedback = None  # 이 부분 주석처리!
                self.min_angle = 999
                self.min_angle_landmarks = None
                self.last_transition_time = now
        elif self.state == 'up':
            if angle > 150:
                self.state = 'down'
                self.count += 1
                count_inc = 1
                self.last_transition_time = now
        return count_inc, None
    def get_min_angle_landmarks(self):
        return self.min_angle_landmarks
    def set_last_min_feedback(self, msg, angle3=None):
        self.last_valid_min_feedback = msg
        if angle3 is not None:
            self.last_angle3_value = angle3
    def get_last_min_feedback(self):
        return self.last_valid_min_feedback
    def get_last_angle3_value(self):
        return self.last_angle3_value

# ========== 기존 REP COUNTER ==========
class RepCounter:
    def __init__(self, angle_thresholds, tempo_thresholds):
        self.angle_up = angle_thresholds['up']
        self.angle_down = angle_thresholds['down']
        self.min_time = tempo_thresholds['min_time']
        self.max_time = tempo_thresholds['max_time']
        self.state = 'init'
        self.count = 0
        self.last_transition_time = None
    def update(self, angle, now=None):
        if now is None:
            now = time.time()
        count_inc = 0
        tempo_msg = None
        if self.state == 'init':
            if angle <= self.angle_down:
                self.state = 'down'
            elif angle >= self.angle_up:
                self.state = 'up'
            self.last_transition_time = now
            return 0, None
        if self.state == 'down':
            if angle >= self.angle_up:
                self.state = 'up'
                self.last_transition_time = now
        elif self.state == 'up':
            if angle <= self.angle_down:
                self.state = 'down'
                self.count += 1
                count_inc = 1
                delta = now - (self.last_transition_time or now)
                tempo_msg = self._evaluate_tempo(delta)
                self.last_transition_time = now
        return count_inc, tempo_msg
    def _evaluate_tempo(self, delta):
        if delta < self.min_time:
            return f'tempo_too_fast ({Utils.format_time(delta)})'
        if delta > self.max_time:
            return f'tempo_too_slow ({Utils.format_time(delta)})'
        return None

# ========== EXERCISE ANALYZER (pullup feedback 추가) ==========
class ExerciseAnalyzer:
    _counters = {}
    @staticmethod
    def analyze_exercise(name, landmarks, now=None):
        feedback_items = []
        if name == 'pullup':
            f1 = f2 = f3 = None
            try:
                left_sh, right_sh = landmarks[11], landmarks[12]
                left_wr, right_wr = landmarks[15], landmarks[16]
                angle1 = Utils.angle_between_lines(left_sh, right_sh, left_wr, right_wr)
                if angle1 <= 10:
                    f1 = "어깨-손목선 평행: 좋음"
                elif angle1 <= 20:
                    f1 = "어깨-손목선 평행: 주의"
                else:
                    f1 = "어깨-손목선 평행: 불균형"
            except:
                f1 = None
            try:
                sh_center = ((landmarks[11][0] + landmarks[12][0]) // 2,
                             (landmarks[11][1] + landmarks[12][1]) // 2)
                hip_center = ((landmarks[23][0] + landmarks[24][0]) // 2,
                              (landmarks[23][1] + landmarks[24][1]) // 2)
                angle2 = Utils.angle_with_vertical(sh_center, hip_center)
                if angle2 <= 10:
                    f2 = "척추선 수직: 좋음"
                elif angle2 <= 20:
                    f2 = "척추선 수직: 주의"
                else:
                    f2 = "척추선 수직: 비뚤어짐"
            except:
                f2 = None
            try:
                counter = ExerciseAnalyzer._counters.get('pullup')
                minlms = None
                if counter is not None and hasattr(counter, 'get_min_angle_landmarks'):
                    minlms = counter.get_min_angle_landmarks()
                if minlms and all(k in minlms and minlms[k] is not None for k in (11, 12, 13, 15)):
                    angle3 = Utils.angle_between_lines(minlms[11], minlms[12], minlms[13], minlms[15])
                    if 80 <= angle3 <= 100:
                        f3 = f"최대수축 손목-팔꿈치/어깨선 직교: 좋음 ({int(angle3)}°)"
                    elif 70 <= angle3 < 80 or 100 < angle3 <= 110:
                        f3 = f"최대수축 손목-팔꿈치/어깨선 직교: 주의 ({int(angle3)}°)"
                    else:
                        f3 = f"최대수축 손목-팔꿈치/어깨선 직교: 미흡 ({int(angle3)}°)"
                    if counter is not None:
                        counter.set_last_min_feedback(f3, angle3)
                else:
                    if counter is not None and counter.get_last_min_feedback():
                        # 직전 피드백은 angle3 값 포함해서 유지
                        last_f3 = counter.get_last_min_feedback()
                        f3 = last_f3
                    else:
                        f3 = None
            except:
                f3 = None
            if f1: feedback_items.append(f1)
            if f2: feedback_items.append(f2)
            if f3: feedback_items.append(f3)
            if name not in ExerciseAnalyzer._counters:
                ExerciseAnalyzer._counters[name] = PullupCounter()
            counter = ExerciseAnalyzer._counters[name]
            count_inc, _ = counter.update(landmarks, now)
            return count_inc, None, None, feedback_items
        return 0, None, None, []
    
# ========== EXERCISE (피드백 추가) ==========
class Exercise:
    def __init__(self, name):
        if name not in EXERCISE_PARAMS:
            raise ValueError(f"Unsupported exercise: {name}")
        self.name = name
        self.params = EXERCISE_PARAMS[name]
        self.total_count = 0
    def update(self, landmarks, now=None):
        # 피드백 추가
        if self.name == 'pullup':
            count_inc, tempo_code, angle_code, feedback_items = ExerciseAnalyzer.analyze_exercise(self.name, landmarks, now)
        else:
            count_inc, tempo_code, angle_code = ExerciseAnalyzer.analyze_exercise(self.name, landmarks, now)
            feedback_items = []
        self.total_count += count_inc
        feedback_list = []
        count_msg = FEEDBACK_MESSAGES['count'].format(exercise=self.name, count=self.total_count)
        feedback_list.append(count_msg)
        if tempo_code:
            key = tempo_code.split()[0]
            tmpl = FEEDBACK_MESSAGES.get(key)
            if tmpl:
                tempo_msg = tmpl.format(min_time=self.params['tempo']['min_time'], max_time=self.params['tempo']['max_time'])
                feedback_list.append(tempo_msg)
            else:
                feedback_list.append("템포")
        if angle_code:
            angle_msg = FEEDBACK_MESSAGES.get(angle_code)
            if angle_msg:
                feedback_list.append(angle_msg)
        feedback_list.extend(feedback_items)  # 피드백 추가
        return self.total_count, feedback_list
# ========== POSE ESTIMATOR ==========
class PoseEstimator:
    mp_pose = mp.solutions.pose
    _pose = mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    @staticmethod
    def get_pose_landmarks(frame):
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = PoseEstimator._pose.process(rgb)
        h, w = frame.shape[:2]
        landmarks = {}
        indices = set(JOINT_INDICES) | set(HEAD_LANDMARKS.values())
        if results.pose_landmarks:
            for idx in indices:
                lm = results.pose_landmarks.landmark[idx]
                if hasattr(lm, 'visibility') and lm.visibility < 0.5:
                    continue
                x, y = int(lm.x * w), int(lm.y * h)
                landmarks[idx] = (x, y)
        return landmarks
    @staticmethod
    def get_head_center(landmarks):
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
    @staticmethod
    def get_shoulder_center(landmarks):
        left_sh = 11
        right_sh = 12
        if left_sh in landmarks and right_sh in landmarks:
            x = (landmarks[left_sh][0] + landmarks[right_sh][0]) // 2
            y = (landmarks[left_sh][1] + landmarks[right_sh][1]) // 2
            return (x, y)
        return None
    @staticmethod
    def get_hip_center(landmarks):
        left_hip = 23
        right_hip = 24
        if left_hip in landmarks and right_hip in landmarks:
            x = (landmarks[left_hip][0] + landmarks[right_hip][0]) // 2
            y = (landmarks[left_hip][1] + landmarks[right_hip][1]) // 2
            return (x, y)
        return None
    @staticmethod
    def estimate_spine(landmarks):
        shoulder = PoseEstimator.get_shoulder_center(landmarks)
        hip      = PoseEstimator.get_hip_center(landmarks)
        if shoulder and hip:
            x = (shoulder[0] + hip[0]) // 2
            y = hip[1]
            return (x, y)
        return None

# ========== VISUALIZER ==========
class Visualizer:
    @staticmethod
    def draw_landmarks(frame, landmarks, scale=0.5, thickness=1):
        # 필요시 점도 알파 처리 가능
        pass
    @staticmethod
    def draw_transparent_line(frame, pt1, pt2, color, thickness=1, alpha=0.5):
        overlay = frame.copy()
        cv.line(overlay, pt1, pt2, color, thickness, cv.LINE_AA)
        cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    @staticmethod
    def draw_connections(frame, landmarks, connections=None, thickness=1, alpha=0.35):
        if connections is None:
            connections = CONNECTIONS_BASIC
        outer_thickness = thickness*2
        inner_thickness = thickness
        for idx1, idx2 in connections:
            if idx1 in landmarks and idx2 in landmarks:
                pt1 = landmarks[idx1]
                pt2 = landmarks[idx2]
                # 투명 검은 외곽
                Visualizer.draw_transparent_line(frame, pt1, pt2, (0,0,0), outer_thickness, alpha)
                # 투명 컬러(빨강, 파랑 등) 내부
                Visualizer.draw_transparent_line(frame, pt1, pt2, (0,0,255), inner_thickness, alpha*0.7+0.2)
    @staticmethod
    def draw_centers(frame, centers, thickness=1, alpha=0.5):
        outer_thickness = thickness*3
        inner_thickness = thickness
        if centers.get('head') and centers.get('spine'):
            pt1 = centers['head']
            pt2 = centers['spine']
            Visualizer.draw_transparent_line(frame, pt1, pt2, (0,0,0), outer_thickness, alpha)
            Visualizer.draw_transparent_line(frame, pt1, pt2, (255,0,0), inner_thickness, alpha*0.7+0.2)
    @staticmethod
    def draw_feedback(frame, feedback_list, scale=0.7, thickness=2):
        img_pil = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font = None
        for path in FONT_PATHS:
            try:
                font = ImageFont.truetype(path, int(18 * scale))
                break
            except IOError:
                continue
        if font is None:
            font = ImageFont.load_default()
        # ---- 텍스트 겹침 방지 (라인 간격, y위치, 박스 간격 조정) ----
        x0 = int(10 * scale)
        y0 = int(10 * scale)
        padding = int(8 * scale)
        line_height = int(24 * scale)
        box_margin = int(8 * scale)  # 박스끼리 세로 간격
        cur_y = y0
        for text in feedback_list:
            # 긴 텍스트 줄바꿈 처리 (최대 38자/줄 기준)
            max_line_len = 38
            lines = [text[j:j+max_line_len] for j in range(0, len(text), max_line_len)]
            total_box_height = len(lines) * line_height
            # 각 feedback마다 충분한 세로 간격을 확보
            box_top = cur_y
            box_bottom = cur_y + total_box_height + padding
            # 박스(배경) 그리기
            max_text_w = 0
            for lidx, ltxt in enumerate(lines):
                try:
                    text_w, text_h = font.getsize(ltxt)
                except AttributeError:
                    bbox = draw.textbbox((x0, cur_y + lidx*line_height), ltxt, font=font)
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]
                max_text_w = max(max_text_w, text_w)
            draw.rectangle([x0 - 5, box_top - 5, x0 + max_text_w + 5, box_bottom + 5], fill=(255, 255, 255, 240))
            # 텍스트 그리기
            for lidx, ltxt in enumerate(lines):
                y = cur_y + (lidx * line_height)
                draw.text((x0, y), ltxt, font=font, fill=(0, 0, 0))
            # 다음 박스 y위치로 이동
            cur_y = box_bottom + box_margin
        frame[:, :] = cv.cvtColor(np.array(img_pil), cv.COLOR_RGB2BGR)

# ========== MAIN PROGRAM ==========

class MainProgram:
    paused = False
    flipped = False
    recording = False
    writer = None
    @staticmethod
    def center_window(root, width, height):
        root.update_idletasks()
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        x = (screen_w // 2) - (width // 2)
        y = (screen_h // 2) - (height // 2)
        root.geometry(f"{width}x{height}+{x}+{y}")
    @staticmethod
    def select_source():
        root = tk.Tk()
        w, h = 300, 100
        MainProgram.center_window(root, w, h)
        root.title("소스 선택")
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
    @staticmethod
    def select_exercise():
        root = tk.Tk()
        w, h = 300, 120
        MainProgram.center_window(root, w, h)
        root.title("운동 선택")
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
    @staticmethod
    def handle_keys(key, frame_size):
        w, h = frame_size
        if key == ord(PAUSE_KEY):
            MainProgram.paused = not MainProgram.paused
        elif key == 27: # ESC 키
            MainProgram.paused = False
            cv.destroyAllWindows()
            exit()    
                    
        elif key == ord(FLIP_KEY):
            MainProgram.flipped = not MainProgram.flipped
        elif key == ord(RECORD_KEY):
            MainProgram.recording = not MainProgram.recording
            if MainProgram.recording:
                fourcc = cv.VideoWriter_fourcc(*'mp4v')
                MainProgram.writer = cv.VideoWriter('output.mp4', fourcc, 30.0, (w, h))
            else:
                if MainProgram.writer is not None:
                    MainProgram.writer.release()
                    MainProgram.writer = None
    @staticmethod
    def run():
        source = MainProgram.select_source()
        cap = cv.VideoCapture(source)
        if not cap.isOpened():
            print(f"Error: cannot open source {source}")
            return
        exercise_name = MainProgram.select_exercise()
        exercise = Exercise(exercise_name)
        width  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if MainProgram.flipped:
                frame = cv.flip(frame, 1)
            landmarks = PoseEstimator.get_pose_landmarks(frame)
            now = time.time()
            count, feedback_list = exercise.update(landmarks, now)
            centers = {
                'head':     PoseEstimator.get_head_center(landmarks),
                'shoulder': PoseEstimator.get_shoulder_center(landmarks),
                'hip':      PoseEstimator.get_hip_center(landmarks),
                'spine':    PoseEstimator.estimate_spine(landmarks)
            }
            Visualizer.draw_landmarks(frame, landmarks, SCALE, THICKNESS)
            Visualizer.draw_connections(frame, landmarks, CONNECTIONS_BASIC, THICKNESS)
            Visualizer.draw_centers(frame, centers, THICKNESS)
            Visualizer.draw_feedback(frame, feedback_list, thickness=THICKNESS)
            if MainProgram.recording and MainProgram.writer:
                MainProgram.writer.write(frame)
            cv.imshow('Pose Tracker', frame)
            key = cv.waitKey(1) & 0xFF
            MainProgram.handle_keys(key, (width, height))
            if key == 27:
                break
            if MainProgram.paused:
                MainProgram.handle_keys(cv.waitKey(), (width, height))
                
        cap.release()
        if MainProgram.writer:
            MainProgram.writer.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    MainProgram.run()
