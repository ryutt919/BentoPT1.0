import torch
import numpy as np
from mmengine.config import Config
from mmaction.apis import init_recognizer

def load_stgcn_model(config_path, checkpoint_path, device='cuda:0'):
    """STGCN 모델을 로드합니다."""
    # config 파일 로드
    cfg = Config.fromfile(config_path)
    
    # 모델 초기화 및 가중치 로드
    model = init_recognizer(cfg, checkpoint_path, device=device)
    model.eval()
    
    return model

def preprocess_skeleton(skeleton_data):
    """스켈레톤 데이터를 모델 입력 형식으로 전처리합니다."""
    # 여기서는 skeleton_data가 (T, V, C) 형태라고 가정
    # T: 시간 steps, V: 관절 수, C: 좌표 차원(x,y)
    
    # 배치 차원 추가 및 텐서 변환
    inputs = torch.from_numpy(skeleton_data).float()
    inputs = inputs.unsqueeze(0)  # (1, T, V, C)
    
    return inputs

def predict_action(model, skeleton_data, device='cuda:0'):
    """스켈레톤 시퀀스에 대해 동작을 예측합니다."""
    with torch.no_grad():
        # 데이터 전처리
        inputs = preprocess_skeleton(skeleton_data)
        inputs = inputs.to(device)
        
        # 예측 수행
        result = model.forward(inputs)
        scores = result['pred_score'].cpu().numpy()
        
        # 최고 확률의 클래스 선택
        pred_class = np.argmax(scores)
        confidence = scores[0][pred_class]
        
    return pred_class, confidence

# 클래스 이름 매핑
CLASS_NAMES = [
    'good',
    'elbow_not_vertical',
    'hand_position_bad',
    'elbow_flare',
    'wrist_bent',
    'shoulder_raise',
    'grip_unstable',
    'spine_imbalance',
    'shoulder_asymmetry',
    'barbell_unbalanced',
    'max_contraction_needed',
    'max_relaxation_needed'
]

if __name__ == "__main__":
    # 설정 및 체크포인트 경로
    config_path = "work_dirs/custom_stgcnpp/custom_config.py"
    checkpoint_path = "work_dirs/custom_stgcnpp/best_acc_top1_epoch_50.pth"  # 실제 체크포인트 경로로 변경하세요
    
    # GPU 사용 가능 여부 확인
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # 모델 로드
    model = load_stgcn_model(config_path, checkpoint_path, device)
    
    # 테스트용 더미 데이터 생성 (실제 데이터로 교체 필요)
    dummy_skeleton = np.random.randn(32, 17, 2)  # (시간, 관절수, 좌표)
    
    # 예측 수행
    pred_class, confidence = predict_action(model, dummy_skeleton, device)
    
    # 결과 출력
    print(f"Predicted class: {CLASS_NAMES[pred_class]}")
    print(f"Confidence: {confidence:.4f}")
