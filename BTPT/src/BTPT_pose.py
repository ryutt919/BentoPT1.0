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
        
        # 런지 동작 클래스 매핑
        self.class_names = {
            -1: "분석 중...",
            0: "판별 불가",
            1: "올바른 자세",
            2: "무릎이 발끝을 넘어감",
            3: "등이 구부러짐",
            4: "발이 불안정함",
            5: "몸통이 흔들림",
            6: "뒷무릎이 바닥에 닿음",
            7: "무게중심이 뒤로 쏠림",
            8: "최대 수축 필요",
            9: "최대 이완 필요"
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

    def _load_stgcn_model(self, config_path: str,
                         checkpoint_path: str,
                         device: str = "cuda:0",
                         topk: int = 5) -> Dict:
        cfg   = Config.fromfile(config_path)
        model = init_recognizer(cfg, checkpoint_path, device=device)
        model.eval()
        return {"model": model, "device": device, "topk": topk}        
        
    def extract_keypoints(self, frame: np.ndarray, 
                         frame_number: int) -> Optional[np.ndarray]:
        """한 프레임에서 키포인트 추출"""
        results = self.pose_model(frame, conf=0.3, verbose=False)[0]  # 낮은 confidence도 허용
        
        if results.keypoints is None:
            return None
            
        kpts = results.keypoints.data.cpu().numpy()
        if len(kpts) == 0:
            return None
            
        keypoints = kpts[0]  # 첫 번째 사람만
        
        # 기본 유효성 검사
        if len(keypoints) != 17:
            return None
            
        # 중앙값 필터로 노이즈 감소
        coords = keypoints[:, :2]  # (V, 2)
        conf = keypoints[:, 2:]    # (V, 1)
        
        # 시간축으로 중앙값 필터링 수행
        if len(self.keypoints_buffer) > 0:
            prev_coords = np.array(self.keypoints_buffer)[-1, :, :2]
            coords = (coords + prev_coords) / 2  # 이전 프레임과 평균
            
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
        """슬라이딩 윈도우로 현재 포즈 시퀀스 분석 (파이프라인 + permute 적용)"""
        # 1) 충분한 버퍼 확인
        if len(self.keypoints_buffer) < self.window_size:
            print(f"[DEBUG] Buffer not full: {len(self.keypoints_buffer)}/{self.window_size}")
            return None

        # 2) 시퀀스 배열화
        seq      = list(self.keypoints_buffer)[-self.window_size:]
        sequence = np.array(seq, dtype=np.float32)       # (T, V, 3)
        print(f"[DEBUG] Raw sequence shape: {sequence.shape}")

        # 3) coords & confidence 분리
        coords = sequence[:, :, :2]                      # (T, V, 2)
        conf   = sequence[:, :, 2:]                      # (T, V, 1)
        score3d = conf[..., 0]                           # (T, V)

        # 4) 파이프라인 입력 dict 구성
        data = {
            'keypoint': coords[np.newaxis, ...],   # (1, T, V, 2)
            'keypoint_score': score3d[np.newaxis, ...],  # (1, T, V)
            'num_person': 1
        }
        print(f"[DEBUG] Before pipeline: keypoint {data['keypoint'].shape}, score {data['keypoint_score'].shape}")

        # 5) FormatGCNInput → PackActionInputs 적용
        packed = self.pipeline(data)
        x = packed['inputs']  # (N, M, T, V, C)
        
        # 6) GPU로 이동
        x = x.to(self.stgcn_context["device"])
        print(f"[DEBUG] After pipeline and device move: inputs.shape : {x.shape}, device: {x.device}")

        try:
            print("[DEBUG] Starting model inference...")
            with torch.no_grad():
                # backbone에 직접 전달
                feats = self.stgcn_context["model"].backbone(x)
                out = self.stgcn_context["model"].cls_head(feats)
    
            # 8) 결과 처리
            prob         = torch.softmax(out[0], dim=-1)
            scores_k, labels_k = torch.topk(prob, k=self.stgcn_context["topk"])
            print("\n[DEBUG] Predictions:")
            for idx, (score, label) in enumerate(zip(scores_k.cpu().numpy(),
                                                     labels_k.cpu().numpy()), 1):
                print(f" {idx}. Class {label}: {score:.3f}")

            return {
                "scores": scores_k.cpu().numpy(),
                "labels": labels_k.cpu().numpy(),
                "feedback": self.feedback_list
            }

        except Exception as e:
            print(f"\n[ERROR] Pipeline failed: {e.__class__.__name__}")
            import traceback; traceback.print_exc()
            if 'x' in locals():
                print("\nFinal tensor state:")
                print(f" - Shape : {x.shape}")
                print(f" - Dtype : {x.dtype}")
                print(f" - Device: {x.device}")
            return None

        
    def update_feedback_list(self, result: Dict):
        """피드백 리스트 업데이트"""
        scores = result["scores"]
        labels = result["labels"]
        
        # 피드백 리스트 초기화
        self.feedback_list = []
        
        # 신뢰도 기반 피드백 필터링
        high_conf_feedbacks = []
        low_conf_feedbacks = []
        
        for label, score in zip(labels, scores):
            label = int(label)
            if label == 1:  # 올바른 자세
                if score >= 0.7:  # 높은 신뢰도로 올바른 자세 감지
                    feedback = {
                        "label": label,
                        "message": "정확한 런지 자세입니다👍",
                        "confidence": float(score),
                        "priority": 1
                    }
                    high_conf_feedbacks.append(feedback)
                continue
                
            # 잘못된 자세에 대한 피드백
            if score >= 0.5:  # 높은 신뢰도
                feedback = {
                    "label": label,
                    "message": self.class_names[label],
                    "confidence": float(score),
                    "priority": 2
                }
                high_conf_feedbacks.append(feedback)
            elif score >= 0.3:  # 낮은 신뢰도
                feedback = {
                    "label": label,
                    "message": self.class_names[label],
                    "confidence": float(score),
                    "priority": 3
                }
                low_conf_feedbacks.append(feedback)
        
        # 우선순위에 따라 피드백 정렬 및 병합
        self.feedback_list = sorted(high_conf_feedbacks + low_conf_feedbacks, 
                                  key=lambda x: (x["priority"], -x["confidence"]))
        
        # 피드백 개수 제한 (최대 3개)
        self.feedback_list = self.feedback_list[:3]
        
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
