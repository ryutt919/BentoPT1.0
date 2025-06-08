import cv2 as cv
import numpy as np
import torch
from ultralytics import YOLO
import os
from collections import deque
from mmengine.config import Config
from mmaction.apis import init_recognizer
from typing import Tuple, List, Optional, Dict
from mmengine.dataset import Compose  
from mmaction.datasets.transforms.formatting import FormatGCNInput, PackActionInputs
from scipy.ndimage import median_filter, gaussian_filter

class PoseEstimator:
    def __init__(self):
        # YOLO 모델 로드
        self.pose_model = YOLO('yolov8x-pose.pt')
        self.pose_model.verbose = False
        
        # ST-GCN 모델 로드
        config_path = "configs/custom_config.py"
        checkpoint_path = "models/lunge/best_acc_top1_epoch_50.pth"
        self.stgcn_context = self._load_stgcn_model(
            config_path, checkpoint_path, topk=3, device="cuda:0"
        )
        
        # COCO 포맷의 17개 키포인트 이름 정의
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # 클래스 매핑 수정 (9개 클래스로)
        self.class_names = {
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
        
        # 피드백 리스트 관리
        self.feedback_list = []
        self.feedback_history = deque(maxlen=5)  # 최근 5개의 피드백 기억
        
        # 슬라이딩 윈도우 파라미터
        self.window_size = 32
        self.stride = 16
        self.buffer_size = 64
        self.keypoints_buffer = deque(maxlen=self.buffer_size)
        self.frame_numbers_buffer = deque(maxlen=self.buffer_size)
        self.frame_counter = 0
        
        self.pipeline = Compose([
            FormatGCNInput(num_person=1),
            PackActionInputs()
        ])
        
        self.prediction_buffer = deque(maxlen=10)
        self.confidence_threshold = 0.7

    def _load_stgcn_model(self, config_path: str, checkpoint_path: str, device: str = "cuda:0", topk: int = 5) -> Dict:
        cfg = Config.fromfile(config_path)
        model = init_recognizer(cfg, checkpoint_path, device=device)
        model.eval()
        
        # 모델 구조 출력
        print("\n[DEBUG] Model Structure:")
        print(model)
        
        # 가중치 통계 출력
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n[DEBUG] Total parameters: {total_params:,}")
        
        return {"model": model, "device": device, "topk": topk}        
        
    def extract_keypoints(self, frame: np.ndarray, frame_number: int) -> Optional[np.ndarray]:
        """한 프레임에서 키포인트 추출 - 메디안/가우시안 필터 적용"""
        results = self.pose_model(frame, conf=0.5, verbose=False)[0]
        
        if results.keypoints is None:
            return None
            
        kpts = results.keypoints.data.cpu().numpy()
        if len(kpts) == 0:
            return None
            
        keypoints = kpts[0]  # 첫 번째 사람만
        
        # 기본 유효성 검사
        if len(keypoints) != 17:
            return None
            
        coords = keypoints[:, :2]  # (V, 2)
        conf = keypoints[:, 2:]    # (V, 1)
        
        # (0,0) 좌표와 낮은 신뢰도 처리
        if len(self.keypoints_buffer) > 0:
            prev_coords = np.array(self.keypoints_buffer)[-1, :, :2]
            zero_mask = np.all(coords == 0, axis=1)
            low_conf_mask = conf[:, 0] < 0.3
            
            # 이전 프레임 좌표로 대체
            coords[zero_mask | low_conf_mask] = prev_coords[zero_mask | low_conf_mask]
        
        # 시간적 필터링 (최근 5 프레임)
        if len(self.keypoints_buffer) >= 5:
            recent_frames = np.array([frame[:, :2] for frame in list(self.keypoints_buffer)[-5:]])
            recent_frames = np.concatenate([recent_frames, coords[np.newaxis, ...]], axis=0)
            
            # 메디안 필터 (시간축)
            filtered_coords = median_filter(recent_frames, size=(3,1,1))[-1]
            
            # 가우시안 필터 (공간축)
            # filtered_coords = gaussian_filter(filtered_coords, sigma=0.5)
            
            coords = filtered_coords
        
        # 필터링된 좌표와 confidence 재결합
        keypoints = np.concatenate([coords, conf], axis=1)
        
        return keypoints
        
    def process_frame(self, frame: np.ndarray, frame_number: int) -> Optional[Dict]:
        """프레임 처리 및 분석 수행"""
        keypoints = self.extract_keypoints(frame, frame_number)
        
        if keypoints is not None:
            self.keypoints_buffer.append(keypoints)
            self.frame_numbers_buffer.append(frame_number)
            
            # 충분한 프레임이 쌓이면 분석 수행
            if len(self.keypoints_buffer) >= self.window_size:
                result = self.analyze_pose()
                if result is not None:
                    self.update_feedback_list(result)
                return result
            
        return None
        
    def analyze_pose(self) -> Optional[Dict]:
        """슬라이딩 윈도우로 현재 포즈 시퀀스 분석"""
        if len(self.keypoints_buffer) < self.window_size:
            return None

        # 시퀀스 배열화
        seq = list(self.keypoints_buffer)[-self.window_size:]
        sequence = np.array(seq, dtype=np.float32)       # (T, V, 3)

        # coords & confidence 분리
        coords = sequence[:, :, :2]                      # (T, V, 2)
        conf = sequence[:, :, 2:]                        # (T, V, 1)
        score3d = conf[..., 0]                          # (T, V)

        # 좌표 정규화 추가
        coords_min = coords.min(axis=(0, 1), keepdims=True)
        coords_max = coords.max(axis=(0, 1), keepdims=True)
        coords_normalized = (coords - coords_min) / (coords_max - coords_min + 1e-8)

        # 파이프라인 입력 dict 구성 - 정규화된 좌표 사용
        data = {
            'keypoint': coords_normalized[np.newaxis, ...],   # (1, T, V, 2)
            'keypoint_score': score3d[np.newaxis, ...],      # (1, T, V)
            'num_person': 1
        }

        # FormatGCNInput → PackActionInputs 적용
        packed = self.pipeline(data)
        x = packed['inputs']  # (N, M, T, V, C)
        
        # GPU로 이동
        x = x.to(self.stgcn_context["device"])

        try:
            with torch.no_grad():
                # backbone에 직접 전달
                feats = self.stgcn_context["model"].backbone(x)
                out = self.stgcn_context["model"].cls_head(feats)
    
                # 결과 처리
                prob = torch.softmax(out[0], dim=-1)
                scores_k, labels_k = torch.topk(prob, k=self.stgcn_context["topk"])
        
                return {
                    "scores": scores_k.cpu().numpy(),
                    "labels": labels_k.cpu().numpy(),
                    "feedback": self.feedback_list
                }

        except Exception as e:
            return None

        
    def update_feedback_list(self, result: Dict):
        """피드백 리스트 업데이트 - 다중 피드백 지원"""
        scores = result["scores"]
        labels = result["labels"]
        
        # 예측 결과 버퍼링 - 모든 상위 결과 저장
        for label, score in zip(labels, scores):
            self.prediction_buffer.append((label, score))
        
        # 버퍼가 충분히 쌓였을 때 분석
        if len(self.prediction_buffer) >= 5:
            # 레이블별 평균 신뢰도 계산
            label_scores = {}
            for label, score in self.prediction_buffer:
                if label not in label_scores:
                    label_scores[label] = []
                label_scores[label].append(score)
            
            # 새로운 피드백 리스트 생성
            new_feedback_list = []
            
            for label, scores in label_scores.items():
                avg_score = np.mean(scores)
                count = len(scores)
                
                # 빈도와 신뢰도 조건 검사
                if (count / len(self.prediction_buffer) >= 0.3 and  # 30% 이상 발생
                    avg_score >= self.confidence_threshold):         # 신뢰도 임계값 이상
                    
                    # "올바른 자세" 피드백은 신뢰도가 매우 높을 때만
                    if int(label) == 1 and avg_score < 0.9:
                        continue
                        
                    new_feedback_list.append({
                        "label": int(label),
                        "message": self.class_names[int(label)],
                        "confidence": float(avg_score),
                        "priority": 2 if label != 1 else 1
                    })
            
            # 우선순위에 따라 정렬 (잘못된 자세가 더 높은 우선순위)
            new_feedback_list.sort(key=lambda x: (-x["priority"], -x["confidence"]))
            
            # 상위 3개 피드백만 유지
            self.feedback_list = new_feedback_list[:3]
        
        # 피드백 히스토리 업데이트
        if self.feedback_list:
            self.feedback_history.append(self.feedback_list)
        
    def get_feedback_history(self) -> List[List[Dict]]:
        """최근 피드백 히스토리 반환"""
        return list(self.feedback_history)
        
    def clear_buffer(self):
        """버퍼 초기화"""
        self.keypoints_buffer.clear()
        self.frame_numbers_buffer.clear()
        self.frame_counter = 0
        self.feedback_list = []
        self.feedback_history.clear()
        
    def get_buffer_size(self):
        """현재 버퍼 크기 반환"""
        return len(self.keypoints_buffer)
