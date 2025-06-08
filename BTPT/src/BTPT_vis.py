import cv2 as cv
import numpy as np
from typing import Dict, Optional

class PoseVisualizer:
    def __init__(self):
        # COCO 포맷의 키포인트 연결 정의 (얼굴 제외)
        self.skeleton_links = [
            (5, 7),   # left_shoulder → left_elbow
            (7, 9),   # left_elbow → left_wrist
            (6, 8),   # right_shoulder → right_elbow
            (8, 10),  # right_elbow → right_wrist
            (5, 11),  # left_shoulder → left_hip
            (6, 12),  # right_shoulder → right_hip
            (11, 13), # left_hip → left_knee
            (13, 15), # left_knee → left_ankle
            (12, 14), # right_hip → right_knee
            (14, 16), # right_knee → right_ankle
            (5, 6),   # left_shoulder → right_shoulder
            (11, 12)  # left_hip → right_hip
        ]
        
        # 컬러 정의 (BGR)
        self.colors = {
            'left': (0, 165, 255),     # 왼쪽 관절 (주황색)
            'right': (0, 255, 0),      # 오른쪽 관절 (녹색)
            'center': (255, 255, 255),  # 중앙 관절 (흰색)
            'skeleton': (255, 255, 255), # 스켈레톤 선 (흰색)
            'feedback_good': (0, 255, 0),   # 좋은 피드백 (녹색)
            'feedback_bad': (0, 0, 255),     # 나쁜 피드백 (빨간색)
            'feedback_warn': (0, 165, 255)   # 경고 피드백 (주황색)
        }
        
        # 클래스별 피드백 메시지
        self.feedback_messages = {
            0: "자세가 좋습니다",
            1: "팔꿈치가 수직이 아닙니다",
            2: "손 위치가 잘못되었습니다",
            3: "팔꿈치가 벌어져 있습니다",
            4: "손목이 구부러져 있습니다",
            5: "어깨가 올라가 있습니다",
            6: "그립이 불안정합니다",
            7: "척추 정렬이 불균형합니다",
            8: "어깨가 비대칭입니다",
            9: "바벨이 기울어져 있습니다",
            10: "최대 수축이 필요합니다",
            11: "최대 이완이 필요합니다"
        }    
    def draw_pose(self, frame: np.ndarray, keypoints: np.ndarray, 
                 result: Optional[Dict] = None) -> np.ndarray:
        """
        프레임에 포즈 키포인트, 스켈레톤, 피드백을 그립니다.
        
        Args:
            frame: 원본 비디오 프레임
            keypoints: shape (17, 3)의 키포인트 배열 (x, y, confidence)
            result: ST-GCN 분석 결과 딕셔너리 (scores, labels, feedback)
        """
        # 오버레이 레이어 생성
        overlay = frame.copy()
        
        # 디버깅: 입력 데이터 확인
        height, width = frame.shape[:2]
        cv.putText(overlay, f"Frame size: {width}x{height}", 
                  (10, height - 60), cv.FONT_HERSHEY_SIMPLEX, 
                  0.5, (255, 255, 255), 1, cv.LINE_AA)
        
        if keypoints is not None:
            valid_points = np.sum(keypoints[:, 2] > 0.5)
            cv.putText(overlay, f"Valid keypoints: {valid_points}/17", 
                      (10, height - 40), cv.FONT_HERSHEY_SIMPLEX, 
                      0.5, (255, 255, 255), 1, cv.LINE_AA)
        
        # 1) 키포인트 그리기
        for i, (x, y, conf) in enumerate(keypoints):
            if conf <= 0.5:  # confidence threshold
                continue
                
            x, y = int(x), int(y)
            
            # 키포인트 색상 결정 (얼굴 제외)
            if i < 5:  # 얼굴 관련 키포인트는 건너뛰기
                continue
            elif i in [5, 7, 9, 11, 13, 15]:  # 왼쪽 키포인트
                color = self.colors['left']
            elif i in [6, 8, 10, 12, 14, 16]:  # 오른쪽 키포인트
                color = self.colors['right']
            else:
                continue
            
            # 키포인트 원 그리기
            cv.circle(overlay, (x, y), 5, color, -1)  # 내부 원
            cv.circle(overlay, (x, y), 5, self.colors['center'], 1)  # 테두리
        
        # 2) 스켈레톤 그리기
        for start_idx, end_idx in self.skeleton_links:
            if keypoints[start_idx][2] > 0.5 and keypoints[end_idx][2] > 0.5:
                start_point = tuple(map(int, keypoints[start_idx][:2]))
                end_point = tuple(map(int, keypoints[end_idx][:2]))
                cv.line(overlay, start_point, end_point, self.colors['skeleton'], 2)
        
        # 3) 피드백 표시
        if result is not None and 'feedback' in result:
            feedback_list = result['feedback']
            
            # 디버깅: 원본 분석 결과 표시
            if 'scores' in result and 'labels' in result:
                scores = result['scores']
                labels = result['labels']
                debug_text = f"Raw predictions: "
                for s, l in zip(scores[:3], labels[:3]):
                    debug_text += f"[{l}:{s:.2f}] "
                cv.putText(overlay, debug_text, (10, height - 20), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
            
            # 피드백이 있는 경우
            if feedback_list:
                y_base = 30  # 기본 y 위치
                
                # 디버깅: 총 피드백 수 표시
                cv.putText(overlay, f"Total feedback: {len(feedback_list)}", 
                          (width - 150, 20), cv.FONT_HERSHEY_SIMPLEX, 
                          0.5, (255, 255, 255), 1, cv.LINE_AA)
                
                for i, feedback in enumerate(feedback_list):
                    message = feedback['message']
                    confidence = feedback['confidence']
                    priority = feedback.get('priority', 0)  # priority 필드가 있다면 가져오기
                    
                    # 신뢰도에 따른 색상 선택
                    if confidence >= 0.7:
                        color = self.colors['feedback_bad']
                    elif confidence >= 0.5:
                        color = self.colors['feedback_warn']
                    else:
                        color = self.colors['feedback_good']
                    
                    # 피드백 메시지 표시
                    y_pos = y_base + (i * 30)
                    text = f"{message} (conf:{confidence:.2f}, pri:{priority})"
                    cv.putText(overlay, text, (10, y_pos), 
                             cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv.LINE_AA)
            else:
                # 피드백이 없으면 "올바른 자세" 메시지 표시
                cv.putText(overlay, "올바른 자세입니다", (10, 30),
                          cv.FONT_HERSHEY_SIMPLEX, 0.7, 
                          self.colors['feedback_good'], 2, cv.LINE_AA)
        
        # 4) 알파 블렌딩으로 오버레이 적용
        alpha = 0.6
        cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        return frame
