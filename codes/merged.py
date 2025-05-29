import time
import math
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
    FONT_PATHS, FEEDBACK_MESSAGES, EXERCISE_PARAMS, 
    CONNECTIONS_LEG
)
import csv
import datetime

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
                self.count += 1  # 올라올 때 count 증가!
                count_inc = 1
                delta = now - (self.last_transition_time or now)
                tempo_msg = self._evaluate_tempo(delta)
                self.last_transition_time = now
        elif self.state == 'up':
            if angle <= self.angle_down:
                self.state = 'down'
                self.last_transition_time = now
        return count_inc, tempo_msg
    def _evaluate_tempo(self, delta):
        if delta < self.min_time:
            return f'tempo_too_fast ({Utils.format_time(delta)})'
        if delta > self.max_time:
            return f'tempo_too_slow ({Utils.format_time(delta)})'
        return None

# ========== EXERCISE ANALYZER (pullup feedback 추가) ==========
"""
ExerciseAnalyzer (정규화·좌우 평균/최대 변위 통합 버전)
=====================================================
▶ 주요 변경점
 1. 모든 거리 기반 임계값을 **어깨폭 대비 비율(%)** 로 계산하여 해상도·거리 편차 제거
 2. **좌·우 평균/최대 변위** 로 카메라 블라인드·운동 비대칭·노이즈 완화
 3. 각 운동별 핵심 피드백 주석을 **한글**로 명시

※ 의존 모듈
   Utils        : 각도·거리 계산 유틸(외부 정의)
   Visualizer   : 피드백 포인트 표시(외부 정의)
   RepCounter   : 반복 카운터(스쿼트·런지·푸시업·컬용)
   PullupCounter: 풀업 전용 카운터(최소 각도 저장 기능 포함)
"""

class ExerciseAnalyzer:
    """운동 자세를 프레임별로 분석하여 피드백·반복수를 반환"""
    def get_facing_direction(lms):
        # mediapipe 기준: 0=코, 7=왼귀, 8=오른귀
        if 0 in lms and (7 in lms or 8 in lms):
            nose_x = lms[0][0]
            # 보이는 귀만 사용 (둘 다 있으면 더 멀리 있는 쪽은 신뢰도 낮음)
            if 7 in lms and 8 in lms:
                # 더 가까운(코와 x좌표 차가 작은) 귀를 사용
                left_dist = abs(nose_x - lms[7][0])
                right_dist = abs(nose_x - lms[8][0])
                ear_x = lms[7][0] if left_dist < right_dist else lms[8][0]
            elif 7 in lms:
                ear_x = lms[7][0]
            else:
                ear_x = lms[8][0]
            diff = nose_x - ear_x
            # 음수면 x-방향(왼쪽), 양수면 x+방향(오른쪽) 바라봄
            return diff
        return 0  # 기본값: 정면
    # ────────── 클래스 상태 변수 ──────────
    _counters = {}
    _prev_biceps_elbow = {"left": None, "right": None}
    _prev_biceps_shoulder = {"left": None, "right": None}
    _lunge_knee_history = {"left": [], "right": []}
    _not_exercising_until = {}  # 운동별로 피드백 일시 중단 타이머
    
    _spine_angle_history = []
    _spine_angle_outlier_until = 0

    # ────────── 임계값(어깨폭 대비 비율) ──────────
    _TH = {
        "hip_imbalance": 0.10,      # 골반 좌우 높이차 10 %↑ 시 경고
        "knee_toe_fwd": 0.07,       # 무릎이 발끝을 7 %↑ 넘어가면 경고
        "shoulder_height": 0.08,    # 어깨 좌우 높이차 8 %↑ 경고
        "wrist_shoulder_y": 0.15,   # 손목‑어깨 y 오프셋 15 %↑ 경고
        "hip_center_line": 0.12,    # 푸시업 골반 sag/rise 12 %↑ 경고
        "joint_sway": 0.05,         # 팔·어깨 흔들림 5 %↑ 경고
    }

    # ────────── 보조 함수 ──────────
    
    def _scale_factor(lms, exercise_name):
        try:
            if exercise_name in ["squat", "lunge"]:
                # 골반(hip)-무릎 거리(좌우 중 큰 값)
                left = np.linalg.norm(np.array(lms[23]) - np.array(lms[25]))
                right = np.linalg.norm(np.array(lms[24]) - np.array(lms[26]))
                leg_len = max(left, right)
                return 100.0 / leg_len if leg_len > 1e-6 else 1e-6
            else:
                # 어깨폭(상체 운동)
                w = np.linalg.norm(np.array(lms[11]) - np.array(lms[12]))
                return 100.0 / w if w > 1e-6 else 1e-6
        except Exception:
            return 1e-6
    # ────────── 메인 분석 함수 ──────────
    _frame_feedback_buffer = []
    _frame_exercise_state = []
    _frame_counter = 0
    _FEEDBACK_INTERVAL = 7

    @staticmethod
    def analyze_exercise(name, lms, now=None):
        fb = []
        S = ExerciseAnalyzer._scale_factor(lms, name) or 1e-9
        if now is None:
            now = time.time()

        # === 영상 시작 후 1초간 피드백 비활성화 ===
        if not hasattr(ExerciseAnalyzer, "_video_start_time"):
            ExerciseAnalyzer._video_start_time = now
        if now - ExerciseAnalyzer._video_start_time < 2.0:
            # 카운트는 정상적으로 동작, 피드백만 비움
            return 0, None, None, []

        # ================================================== 1. PULL‑UP
        if name == "pullup":
            # (1) 어깨‑손목선 평행도
            try:
                l_sh, r_sh = lms[11], lms[12]
                l_wr, r_wr = lms[15], lms[16]
                a1 = Utils.angle_between_lines(l_sh, r_sh, l_wr, r_wr)
                fb.append("어깨‑손목선 평행: " + ("좋음" if a1 <= 10 else "주의" if a1 <= 20 else "불균형"))
            except Exception:
                pass
            # (2) 척추 수직도
            try:
                sh_c = ((lms[11][0] + lms[12][0]) // 2, (lms[11][1] + lms[12][1]) // 2)
                hip_c = ((lms[23][0] + lms[24][0]) // 2, (lms[23][1] + lms[24][1]) // 2)
                a2 = Utils.angle_with_vertical(sh_c, hip_c)
                fb.append("척추선 수직: " + ("좋음" if a2 <= 10 else "주의" if a2 <= 20 else "비뚤어짐"))
            except Exception:
                pass
            # (3) 최대 수축 시 전완‑어깨 직교(좌우 평균)
            try:
                counter = ExerciseAnalyzer._counters.get("pullup")
                mm = counter.get_min_angle_landmarks() if counter and hasattr(counter, "get_min_angle_landmarks") else None
                if mm and all(k in mm and mm[k] is not None for k in (11, 12, 13, 14, 15, 16)):
                    ang_l = Utils.angle_between_lines(mm[11], mm[12], mm[13], mm[15])
                    ang_r = Utils.angle_between_lines(mm[11], mm[12], mm[14], mm[16])
                    ang_m = (ang_l + ang_r) / 2
                    grade = "좋음" if 80 <= ang_m <= 100 else "주의" if 70 <= ang_m < 80 or 100 < ang_m <= 110 else "미흡"
                    msg = f"최대수축 전완‑어깨 직교: {grade} ({int(ang_m)}°)"
                    fb.append(msg)
                    if counter:
                        counter.set_last_min_feedback(msg, ang_m)
                elif counter and counter.get_last_min_feedback():
                    fb.append(counter.get_last_min_feedback())
            except Exception:
                pass
            # 카운터 업데이트 및 반환
            if "pullup" not in ExerciseAnalyzer._counters:
                ExerciseAnalyzer._counters["pullup"] = PullupCounter()
            inc, _ = ExerciseAnalyzer._counters["pullup"].update(lms, now)
            return inc, None, None, fb

        # ================================================== 2. SQUAT
        if name == "squat":
            try:
                lh, rh = lms[23], lms[24]  # 양쪽 골반
                lk, rk = lms[25], lms[26]  # 양쪽 무릎
                la, ra = lms[27], lms[28]  # 양쪽 발목
                lt, rt = lms[31], lms[32]  # 양쪽 발끝
                # (1) 무릎 전방 이동 – 좌우 중 최대값 비교
                fwd_l = (lk[0] - lt[0]) * S
                fwd_r = (rk[0] - rt[0]) * S
                if max(fwd_l, fwd_r) > ExerciseAnalyzer._TH["knee_toe_fwd"]:
                    side = "왼쪽" if fwd_l > fwd_r else "오른쪽"
                    fb.append(f"{side} 무릎이 발끝을 7 % 이상 넘음: 주의")
                    Visualizer.add_bad_pose_point(lk if side == "왼쪽" else rk)
                # (2) 골반 좌우 높이차
                if abs(lh[1] - rh[1]) * S > ExerciseAnalyzer._TH["hip_imbalance"]:
                    fb.append("골반 좌우 높이 차이 10 % 이상: 주의")
                    Visualizer.add_bad_pose_point(lh); Visualizer.add_bad_pose_point(rh)
                # (3) 상체 기울기
                sh_c = ((lms[11][0] + lms[12][0]) // 2, (lms[11][1] + lms[12][1]) // 2)
                hip_c = ((lh[0] + rh[0]) // 2, (lh[1] + rh[1]) // 2)
                if Utils.angle_with_vertical(sh_c, hip_c) > 20:
                    fb.append("상체가 너무 숙여짐: 주의")
                    Visualizer.add_bad_pose_point(sh_c)
                # 평균 무릎 각도로 카운트 판정
                knee_ang = (Utils.get_angle(lh, lk, la) + Utils.get_angle(rh, rk, ra)) / 2
            except Exception:
                knee_ang = 180  # 예외 시 기본값
            if "squat" not in ExerciseAnalyzer._counters:
                ExerciseAnalyzer._counters["squat"] = RepCounter({'up': 170, 'down': 90}, {'min_time': 0.5, 'max_time': 3.0})
            inc, tempo = ExerciseAnalyzer._counters["squat"].update(knee_ang, now)
            return inc, tempo, None, fb

        # ================================================== 3. LUNGE
        if name == "lunge":  # 운동 이름이 런지일 때
            try:
                lh, rh = lms[23], lms[24]  # 왼/오 골반 좌표
                lk, rk = lms[25], lms[26]  # 왼/오 무릎 좌표
                la, ra = lms[27], lms[28]  # 왼/오 발목 좌표
                lt, rt = lms[31], lms[32]  # 왼/오 발끝 좌표
        
                facing = ExerciseAnalyzer.get_facing_direction(lms)  # 시선 방향(좌/우) 판별
        
                # 앞발/뒷발 판별 (시선 방향에 따라)
                if facing < 0:  # 왼쪽 바라볼 때
                    if lt[0] < rt[0]:  # 왼발이 앞
                        front_knee, front_ankle, front_hip, front_toe = lk, la, lh, lt
                        back_knee, back_ankle, back_hip, back_toe = rk, ra, rh, rt
                        side = "왼쪽"
                    else:  # 오른발이 앞
                        front_knee, front_ankle, front_hip, front_toe = rk, ra, rh, rt
                        back_knee, back_ankle, back_hip, back_toe = lk, la, lh, lt
                        side = "오른쪽"
                else:  # 오른쪽 바라볼 때
                    if lt[0] > rt[0]:  # 왼발이 앞
                        front_knee, front_ankle, front_hip, front_toe = lk, la, lh, lt
                        back_knee, back_ankle, back_hip, back_toe = rk, ra, rh, rt
                        side = "왼쪽"
                    else:  # 오른발이 앞
                        front_knee, front_ankle, front_hip, front_toe = rk, ra, rh, rt
                        back_knee, back_ankle, back_hip, back_toe = lk, la, lh, lt
                        side = "오른쪽"
        
                # === 발 간격 체크 ===
                foot_gap = abs(lt[0] - rt[0])  # 발끝 간 x좌표 차이
                hip_gap = abs(lh[0] - rh[0])   # 골반 간 x좌표 차이
                if feedback_frame_count % 7 == 0:  # 7프레임마다 로그 출력
                    print(f"발 간격: {foot_gap * S:.2f} % (골반 간격: {hip_gap * S:.2f} %)")
        
                # 발 간격이 너무 좁으면 운동 중 아님 처리
                if foot_gap < hip_gap * 1.3:
                    print(f"발 간격이 너무 좁음: {foot_gap :.2f} % < {hip_gap * 1.3:.2f} %")
                    fb.append("발 간격이 너무 좁음: 운동 중이 아닙니다.")
                    ExerciseAnalyzer._not_exercising_until[name] = now + 1.0  # 1초간 운동 중 아님
                    if hasattr(ExerciseAnalyzer, "_spine_angle_feedback_frame_count"):
                        ExerciseAnalyzer._spine_angle_feedback_frame_count = 0  # 척추각 피드백 카운터 초기화
                    if hasattr(ExerciseAnalyzer, "_spine_angle_last_feedback_frame"):
                        ExerciseAnalyzer._spine_angle_last_feedback_frame = -10  # 척추각 피드백 프레임 초기화
                    return 0, None, None, fb  # 피드백 반환 후 함수 종료
        
                # 운동 중이 아님 상태에서 1초 이내면 피드백 중단
                until = ExerciseAnalyzer._not_exercising_until.get(name, 0)
                if now < until:
                    return 0, None, None, []
        
                # 무릎이 발끝을 넘었는지(앞발 기준) 체크
                if (front_knee[0] - front_toe[0]) * (1 if facing < 0 else -1) < 0:
                    fb.append(f"{side} 무릎이 발끝을 넘음: 주의")
                    Visualizer.add_bad_pose_point(front_knee)
        
                # === 척추-앞발 수직선 각도 기반 무게중심 피드백 ===
                sh_c = PoseEstimator.get_shoulder_center(lms)  # 어깨 중심 좌표
                hip_c = PoseEstimator.get_hip_center(lms)      # 골반 중심 좌표
                if sh_c and hip_c:
                    # 척추 벡터 계산 (어깨중점 → 골반중점)
                    spine_vec = np.array([hip_c[0] - sh_c[0], hip_c[1] - sh_c[1]])
                    vertical_vec = np.array([0, 1])  # 화면 수직 벡터
                    norm_spine = np.linalg.norm(spine_vec)
                    if norm_spine > 1e-6:
                        cos_val = np.dot(spine_vec, vertical_vec) / (norm_spine * np.linalg.norm(vertical_vec))
                        cos_val = np.clip(cos_val, -1.0, 1.0)
                        angle_deg = np.degrees(np.arccos(cos_val))  # 척추-수직선 각도(도)
        
                        # === 이동평균 및 outlier 처리 ===
                        history = ExerciseAnalyzer._spine_angle_history  # 척추각 히스토리
                        outlier_until = ExerciseAnalyzer._spine_angle_outlier_until  # outlier 무시 기간
        
                        # 히트맵 프레임 카운터 및 마지막 피드백 프레임
                        if not hasattr(ExerciseAnalyzer, "_spine_angle_feedback_frame_count"):
                            ExerciseAnalyzer._spine_angle_feedback_frame_count = 0
                        if not hasattr(ExerciseAnalyzer, "_spine_angle_last_feedback_frame"):
                            ExerciseAnalyzer._spine_angle_last_feedback_frame = -10
                        feedback_frame_count = ExerciseAnalyzer._spine_angle_feedback_frame_count
                        last_feedback_frame = ExerciseAnalyzer._spine_angle_last_feedback_frame
        
                        # 최근 7프레임 내에서 30도 이상 차이 발생 시 outlier 처리
                        is_outlier = False
                        if len(history) >= 6:  # 7프레임 이상 쌓였을 때만 검사
                            min_angle = min(history + [angle_deg])
                            max_angle = max(history + [angle_deg])
                            if abs(max_angle - min_angle) > 30:
                                ExerciseAnalyzer._spine_angle_outlier_until = len(history) + 5  # 5프레임 outlier 무시
                                is_outlier = True
        
                        # outlier 기간이면 피드백 X, 아니면 피드백 생성
                        if len(history) < ExerciseAnalyzer._spine_angle_outlier_until:
                            history.append(angle_deg)
                            if len(history) > 10:
                                history.pop(0)
                        else:
                            history.append(angle_deg)
                            if len(history) > 10:
                                history.pop(0)
                            avg_angle = sum(history) / len(history)
                            if feedback_frame_count % 7 == 0:
                                print(f"평균 척추각: {avg_angle:.2f}° (현재: {angle_deg:.2f}°)")
                            # 0~85도만 정상, 나머지는 무게중심 뒤로 경고
                            if not (5 <= avg_angle <= 90):
                                fb.append(f"{side} 무게중심이 뒤에 있음: 주의 (척추각 {int(avg_angle)}°)")
                                # 히트맵 포인트 추가 조건
                                if (feedback_frame_count == 0 or feedback_frame_count - last_feedback_frame >= 5) and not is_outlier:
                                    Visualizer.add_bad_pose_point(hip_c)
                                    ExerciseAnalyzer._spine_angle_last_feedback_frame = feedback_frame_count
                            ExerciseAnalyzer._spine_angle_feedback_frame_count = feedback_frame_count + 1
        
                # 뒷발 무릎 각도 계산 (카운트 기준)
                knee_ang = Utils.get_angle(back_hip, back_knee, back_ankle)
        
            except Exception:
                knee_ang = 180  # 예외 발생 시 기본값
            if "lunge" not in ExerciseAnalyzer._counters:
                ExerciseAnalyzer._counters["lunge"] = RepCounter({'up': 150, 'down': 100}, {'min_time': 0.5, 'max_time': 3.0})
            inc, tempo = ExerciseAnalyzer._counters["lunge"].update(knee_ang, now)
            return inc, tempo, None, fb  # 카운트, 템포, 각도코드(None), 피드백 반환
#=============== 4. BICEPS CURL ====================
        if name == "biceps_curl":
            try:
                # 좌·우 어깨·팔꿈치·손목 좌표
                pts = {
                    "left":  (lms[11], lms[13], lms[15]),
                    "right": (lms[12], lms[14], lms[16])
                }
                elbow_angles = []

                for side, (sh, el, wr) in pts.items():
                    # 팔꿈치 각도 계산
                    ang = Utils.get_angle(sh, el, wr)
                    elbow_angles.append(ang)

                    # 팔꿈치 위치 변동(흔들림) 체크
                    prev_el = ExerciseAnalyzer._prev_biceps_elbow[side]
                    if prev_el is not None:
                        move_ratio = np.linalg.norm(np.array(el) - np.array(prev_el)) * S
                        if move_ratio > ExerciseAnalyzer._TH["joint_sway"]:
                            fb.append(f"{side} 팔꿈치가 5 % 이상 이동: 고정 필요")
                            Visualizer.add_bad_pose_point(el)
                    ExerciseAnalyzer._prev_biceps_elbow[side] = el

                    # 어깨 위치 변동(흔들림) 체크
                    prev_sh = ExerciseAnalyzer._prev_biceps_shoulder[side]
                    if prev_sh is not None:
                        sh_move = np.linalg.norm(np.array(sh) - np.array(prev_sh)) * S
                        if sh_move > ExerciseAnalyzer._TH["joint_sway"]:
                            fb.append(f"상체가 5 % 이상 흔들림({side}): 고정 필요")
                            Visualizer.add_bad_pose_point(sh)
                    ExerciseAnalyzer._prev_biceps_shoulder[side] = sh

                    # 팔꿈치 각도 피드백
                    if ang < 40:
                        fb.append(f"{side} 팔꿈치 각도: 너무 굽힘")
                        Visualizer.add_bad_pose_point(el)
                    elif ang > 160:
                        fb.append(f"{side} 팔꿈치 각도: 너무 펴짐")
                        Visualizer.add_bad_pose_point(el)
                # 두 팔 평균 각도로 카운트 판정
                avg_ang = sum(elbow_angles) / len(elbow_angles)
            except Exception:
                avg_ang = 180
            if "biceps_curl" not in ExerciseAnalyzer._counters:
                ExerciseAnalyzer._counters["biceps_curl"] = RepCounter({'up': 40, 'down': 160}, {'min_time': 1.0, 'max_time': 3.0})
            inc, tempo = ExerciseAnalyzer._counters["biceps_curl"].update(avg_ang, now)
            return inc, tempo, None, fb

        #=============== 5. PUSH-UP ========================
        if name == "pushup":
            try:
                # 주요 키포인트 추출
                l_sh, l_el, l_wr = lms[11], lms[13], lms[15]
                r_sh, r_el, r_wr = lms[12], lms[14], lms[16]
                l_hip, r_hip = lms[23], lms[24]
                l_ank, r_ank = lms[27], lms[28]

                # (1) 어깨 좌우 높이차
                if abs(l_sh[1] - r_sh[1]) * S > ExerciseAnalyzer._TH["shoulder_height"]:
                    fb.append("어깨 높이 차이 8 % 이상: 주의")
                    Visualizer.add_bad_pose_point(l_sh); Visualizer.add_bad_pose_point(r_sh)

                # (2) 손목 위치 – 어깨 y좌표 차이
                for side, wr, sh in (("왼", l_wr, l_sh), ("오른", r_wr, r_sh)):
                    if abs(wr[1] - sh[1]) * S > ExerciseAnalyzer._TH["wrist_shoulder_y"]:
                        fb.append(f"{side}손 위치가 어깨보다 15 % 이상 앞/뒤: 주의")
                        Visualizer.add_bad_pose_point(wr)

                # (3) 골반 sag/rise
                hip_c = ((l_hip[0] + r_hip[0]) // 2, (l_hip[1] + r_hip[1]) // 2)
                mid_y = (l_sh[1] + r_sh[1] + l_ank[1] + r_ank[1]) // 4
                if abs(hip_c[1] - mid_y) * S > ExerciseAnalyzer._TH["hip_center_line"]:
                    fb.append("엉덩이 들림/처짐 12 % 이상: 주의")
                    Visualizer.add_bad_pose_point(hip_c)

                # 평균 팔꿈치 각도로 카운트
                ang = (Utils.get_angle(l_sh, l_el, l_wr) + Utils.get_angle(r_sh, r_el, r_wr)) / 2
            except Exception:
                ang = 180
            if "pushup" not in ExerciseAnalyzer._counters:
                ExerciseAnalyzer._counters["pushup"] = RepCounter({'up': 170, 'down': 80}, {'min_time': 0.5, 'max_time': 3.0})
            inc, tempo = ExerciseAnalyzer._counters["pushup"].update(ang, now)
            return inc, tempo, None, fb
        return 0, None, None, []
    
# ========== EXERCISE (피드백 추가) ==========
class Exercise:
    def __init__(self, name):
        if name not in EXERCISE_PARAMS:
            raise ValueError(f"Unsupported exercise: {name}")
        self.name = name
        self.params = EXERCISE_PARAMS[name]
        self.total_count = 0
        self._last_feedback_list = []  # 마지막 피드백 메시지 저장

    def _make_feedback_list(self, count_inc, tempo_code, angle_code, feedback_items):
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
        feedback_list.extend(feedback_items)
        return feedback_list

    def update(self, landmarks, now=None):
        count_inc, tempo_code, angle_code, feedback_items, = ExerciseAnalyzer.analyze_exercise(self.name, landmarks, now)
        self.total_count += count_inc
        feedback_list = self._make_feedback_list(count_inc, tempo_code, angle_code, feedback_items)
        # 메시지가 없다면 이전 메시지 유지
        if len(feedback_list) <= 1:  # 카운트 메시지만 있을 때
            feedback_list = self._last_feedback_list or feedback_list
        else:
            self._last_feedback_list = feedback_list
        return self.total_count, feedback_list,
# =========== 
class OneEuroFilter:
    def __init__(self, freq=30, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0
        self.last_time = None

    def alpha(self, cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def filter(self, x, t=None):
        if self.x_prev is None:
            self.x_prev = x
            return x
        dx = (x - self.x_prev) * self.freq
        dx_hat = self.dx_prev + self.alpha(self.d_cutoff) * (dx - self.dx_prev)
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        alpha = self.alpha(cutoff)
        x_hat = self.x_prev + alpha * (x - self.x_prev)
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat

# ========== POSE ESTIMATOR ==========
class PoseEstimator:
    mp_pose = mp.solutions.pose
    _pose = mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    _filters = {}  # One Euro Filter 인스턴스 저장용

    @staticmethod
    def get_pose_landmarks(frame, min_cutoff=0.2, beta=0.15):
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
                # --- One Euro Filter 적용 --- ## Robust한 선 움직임임
               
                if idx not in PoseEstimator._filters:
                    PoseEstimator._filters[idx] = (
                        OneEuroFilter(min_cutoff, beta), #min_cutoff: 0.2 ~ 0.5 (작을수록 빠름, 노이즈는 약간 증가)
                        OneEuroFilter(min_cutoff, beta) #beta: 0.05 ~ 0.2 (클수록 빠른 움직임에 더 잘 반응)
                    )
                fx, fy = PoseEstimator._filters[idx]
                x_f = int(fx.filter(x))
                y_f = int(fy.filter(y))
                landmarks[idx] = (x_f, y_f)
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
            x = hip[0]
            y = hip[1]
            return (x, y)
        return None

# ========== VISUALIZER ==========
class Visualizer:
    _bad_pose_points = []  # ← 클래스 변수로 선언

    @staticmethod
    def add_bad_pose_point(point):
        Visualizer._bad_pose_points.append(point)
        if len(Visualizer._bad_pose_points) > 200:
            Visualizer._bad_pose_points.pop(0)

    @staticmethod
    def draw_bad_pose_heatmap(frame):
        for pt in Visualizer._bad_pose_points:
            cv.circle(frame, pt, 3, (0,255,255), -1)
    @staticmethod
    def draw_perpendicular_lines(frame, landmarks, exercise_name=None, length=60, color=(0,255,0), thickness=1, alpha=0.5):
        if exercise_name in ['squat', 'lunge']:
            # 하체 운동: 발가락 끝에서 무릎 y까지 수직선, 두 쪽 중 더 긴 길이로 통일
            required_keys = [25, 26, 31, 32]  # 무릎, 발가락 끝
            if not all(idx in landmarks for idx in required_keys):
                return
            # 각 다리의 길이 계산
            left_len = abs(landmarks[31][1] - landmarks[25][1])
            right_len = abs(landmarks[32][1] - landmarks[26][1])
            max_len = max(left_len, right_len)
            # 양쪽 모두 max_len으로 선 그리기
            for toe_idx in [31, 32]:
                this_toe = landmarks[toe_idx]
                pt1 = (int(this_toe[0]), int(this_toe[1]))
                pt2 = (int(this_toe[0]), int(this_toe[1] - max_len))  # 위로 max_len만큼
                Visualizer.draw_transparent_line(frame, pt1, pt2, color, thickness*2, alpha)
        else:
            # 상체 운동: 손목-팔꿈치 기준 (기존 코드)
            required_keys = [13, 14, 15, 16]
            if not all(idx in landmarks for idx in required_keys):
                return
            wrist_a = landmarks[15]
            wrist_b = landmarks[16]
            for wrist_idx, elbow_idx in [(15, 13), (16, 14)]:
                this_wrist = landmarks[wrist_idx]
                this_elbow = landmarks[elbow_idx]
                ab = np.array([wrist_b[0] - wrist_a[0], wrist_b[1] - wrist_a[1]], dtype=float)
                perp = np.array([-ab[1], ab[0]])
                perp_norm = perp / (np.linalg.norm(perp) + 1e-8)
                dir_we = np.array([this_elbow[0] - this_wrist[0], this_elbow[1] - this_wrist[1]], dtype=float)
                if np.dot(perp_norm, dir_we) < 0:
                    perp_norm = -perp_norm
                pt1 = (int(this_wrist[0]), int(this_wrist[1]))
                pt2 = (int(this_wrist[0] + perp_norm[0] * length), int(this_wrist[1] + perp_norm[1] * length))
                Visualizer.draw_transparent_line(frame, pt1, pt2, color, thickness, alpha)

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
        # 머리, 어깨, 힙 중점이 모두 있을 때만 그리기
        if centers.get('head') and centers.get('shoulder') and centers.get('hip'):
            pt_head = centers['head']
            pt_shoulder = centers['shoulder']
            pt_hip = centers['hip']
            # 머리-어깨-힙을 순서대로 잇기
            Visualizer.draw_transparent_line(frame, pt_head, pt_shoulder, (0,0,0), outer_thickness, alpha)
            Visualizer.draw_transparent_line(frame, pt_shoulder, pt_hip, (0,0,0), outer_thickness, alpha)
            Visualizer.draw_transparent_line(frame, pt_head, pt_shoulder, (255,0,0), inner_thickness, alpha*0.7+0.2)
            Visualizer.draw_transparent_line(frame, pt_shoulder, pt_hip, (255,0,0), inner_thickness, alpha*0.7+0.2)
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
    def save_record(exercise_name, count):
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("exercise_log.csv", "a", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([now, exercise_name, count])
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

        target_w, target_h = int(1920/1.5), int(1080/1.5)
        aspect = width / height
        if aspect >= 16/9:
            disp_width = target_w
            disp_height = int(target_w / aspect)
        else:
            disp_height = target_h
            disp_width = int(target_h * aspect)

        # 화면 중앙 위치 계산 (tkinter 사용)
        root = tk.Tk()
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        root.destroy()
        win_x = (screen_w - disp_width) // 2
        win_y = (screen_h - disp_height) // 2

        cv.namedWindow('Pose Tracker', cv.WINDOW_NORMAL)
        cv.resizeWindow('Pose Tracker', disp_width, disp_height)
        cv.moveWindow('Pose Tracker', win_x, win_y)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if MainProgram.flipped:
                frame = cv.flip(frame, 1)

            # 프레임 리사이즈
            frame = cv.resize(frame, (disp_width, disp_height))

        # === 가우시안 블러 적용 ===
            frame_blur = cv.GaussianBlur(frame, (5, 5), 0)

            # 관절 추출 (블러 적용된 프레임 사용)
            landmarks = PoseEstimator.get_pose_landmarks(frame_blur)            
            now = time.time()
            count, feedback_list = exercise.update(landmarks, now)
            centers = {
                'head':     PoseEstimator.get_head_center(landmarks),
                'shoulder': PoseEstimator.get_shoulder_center(landmarks),
                'hip':      PoseEstimator.get_hip_center(landmarks),
                'spine':    PoseEstimator.estimate_spine(landmarks)
            }
            Visualizer.draw_connections(frame, landmarks, CONNECTIONS_BASIC, THICKNESS)
            Visualizer.draw_perpendicular_lines(frame, landmarks, exercise_name=exercise_name)
            Visualizer.draw_centers(frame, centers, THICKNESS)
            Visualizer.draw_feedback(frame, feedback_list, thickness=THICKNESS)
            Visualizer.draw_bad_pose_heatmap(frame)
            if exercise_name == 'squat' or exercise_name == 'pushup' or exercise_name == 'lunge' :
                Visualizer.draw_connections(frame, landmarks, CONNECTIONS_LEG, THICKNESS)
            if MainProgram.recording and MainProgram.writer:
                MainProgram.writer.write(frame)
            cv.imshow('Pose Tracker', frame)
            key = cv.waitKey(1) & 0xFF

            # === a/d로 앞뒤 이동 기능 추가 ===
            if key in [ord('a'), ord('A')]:  # a: 뒤로 1.5초
                fps = cap.get(cv.CAP_PROP_FPS) or 30
                cur = cap.get(cv.CAP_PROP_POS_FRAMES)
                cap.set(cv.CAP_PROP_POS_FRAMES, max(0, cur - int(fps * 1.5)))
                continue
            elif key in [ord('d'), ord('D')]:  # d: 앞으로 1.5초
                fps = cap.get(cv.CAP_PROP_FPS) or 30
                cur = cap.get(cv.CAP_PROP_POS_FRAMES)
                total = cap.get(cv.CAP_PROP_FRAME_COUNT)
                cap.set(cv.CAP_PROP_POS_FRAMES, min(total-1, cur + int(fps * 1.5)))
                continue

            MainProgram.handle_keys(key, (disp_width, disp_height))
            if key == 27:
                break
            if MainProgram.paused:
                MainProgram.handle_keys(cv.waitKey(), (disp_width, disp_height))
                
        cap.release()
        if MainProgram.writer:
            MainProgram.writer.release()
        cv.destroyAllWindows()
        MainProgram.save_record(exercise_name, exercise.total_count)  
    
    
              

if __name__ == "__main__":
    MainProgram.run()
