import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
from mmaction.apis import init_recognizer
from mmengine.config import Config, ConfigDict

class NpzSkeletonDataset(Dataset):
    """NPZ 파일에서 스켈레톤 시퀀스를 로드하는 데이터셋"""
    def __init__(self, npz_dir, window_size=32, stride=16):
        self.npz_paths = list(Path(npz_dir).glob('*.npz'))
        self.window_size = window_size
        self.stride = stride
        
        # COCO 키포인트 매핑 정의
        self.YOLO_TO_COCO = {
            0: 0,    # nose -> nose
            5: 5,    # left_shoulder -> left_shoulder
            6: 6,    # right_shoulder -> right_shoulder
            7: 7,    # left_elbow -> left_elbow
            8: 8,    # right_elbow -> right_elbow
            9: 9,    # left_wrist -> left_wrist
            10: 10,  # right_wrist -> right_wrist
            11: 11,  # left_hip -> left_hip
            12: 12,  # right_hip -> right_hip
            13: 13,  # left_knee -> left_knee
            14: 14,  # right_knee -> right_knee
            15: 15,  # left_ankle -> left_ankle
            16: 16   # right_ankle -> right_ankle
        }
    
    def __len__(self):
        return len(self.npz_paths)
    
    def _convert_to_coco_format(self, keypoints, frame_numbers):
        """YOLOv8 키포인트를 COCO 포맷으로 변환"""
        T = len(frame_numbers)
        coco_kpts = np.zeros((T, 17, 3))  # 17개의 COCO 키포인트
        
        # 있는 키포인트 복사
        for yolo_idx, coco_idx in self.YOLO_TO_COCO.items():
            if yolo_idx < keypoints.shape[1]:
                coco_kpts[:, coco_idx] = keypoints[:, yolo_idx]
        
        # 없는 키포인트 추정 (눈, 귀)
        nose_idx, left_shoulder_idx, right_shoulder_idx = 0, 5, 6
        
        for t in range(T):
            if coco_kpts[t, nose_idx, 2] > 0 and coco_kpts[t, left_shoulder_idx, 2] > 0:
                # 왼쪽 눈과 귀
                coco_kpts[t, 1] = 0.7 * coco_kpts[t, nose_idx] + 0.3 * coco_kpts[t, left_shoulder_idx]
                coco_kpts[t, 3] = 0.3 * coco_kpts[t, nose_idx] + 0.7 * coco_kpts[t, left_shoulder_idx]
                coco_kpts[t, [1, 3], 2] = min(coco_kpts[t, nose_idx, 2], coco_kpts[t, left_shoulder_idx, 2])
            
            if coco_kpts[t, nose_idx, 2] > 0 and coco_kpts[t, right_shoulder_idx, 2] > 0:
                # 오른쪽 눈과 귀
                coco_kpts[t, 2] = 0.7 * coco_kpts[t, nose_idx] + 0.3 * coco_kpts[t, right_shoulder_idx]
                coco_kpts[t, 4] = 0.3 * coco_kpts[t, nose_idx] + 0.7 * coco_kpts[t, right_shoulder_idx]
                coco_kpts[t, [2, 4], 2] = min(coco_kpts[t, nose_idx, 2], coco_kpts[t, right_shoulder_idx, 2])
        
        return coco_kpts[:, :, :2], coco_kpts[:, :, 2]  # 키포인트와 confidence score 분리
    
    def __getitem__(self, idx):
        npz_path = self.npz_paths[idx]
        data = np.load(npz_path, allow_pickle=True)
        
        # 데이터 로드
        keypoints = data['keypoints']  # (T, V, 3)
        frame_numbers = data['frame_numbers']
        
        # COCO 포맷으로 변환
        kp, kpscore = self._convert_to_coco_format(keypoints, frame_numbers)
        
        # 시퀀스가 너무 짧으면 패딩
        if len(frame_numbers) < self.window_size:
            pad_size = self.window_size - len(frame_numbers)
            kp = np.pad(kp, ((0, pad_size), (0, 0), (0, 0)), mode='edge')
            kpscore = np.pad(kpscore, ((0, pad_size), (0, 0)), mode='edge')
        
        # 첫 번째 윈도우만 사용 (또는 랜덤하게 선택 가능)
        start_idx = 0
        end_idx = start_idx + self.window_size
        
        # 윈도우 선택
        window_keypoints = kp[start_idx:end_idx]  # (T, V, 2)
        window_kpscore = kpscore[start_idx:end_idx]  # (T, V)
        
        # MMAction2 입력 형식으로 변환
        # (T, V, 2) -> (2, T, V) -> (1, 2, T, V, 1)
        x = torch.tensor(window_keypoints, dtype=torch.float32)
        x = x.permute(2, 0, 1)  # (T, V, 2) -> (2, T, V)
        x = x.unsqueeze(0).unsqueeze(-1)  # (2, T, V) -> (1, 2, T, V, 1)
        
        # confidence score도 같은 형식으로 변환
        s = torch.tensor(window_kpscore, dtype=torch.float32)
        s = s.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # (T, V) -> (1, 1, T, V, 1)
        
        # 키포인트와 confidence score 결합
        x = torch.cat([x, s], dim=1)  # (1, 3, T, V, 1)
        
        return {
            'keypoints': x,
            'path': str(npz_path)
        }

def extract_features(model, dataloader, device):
    """ST-GCN 모델을 사용하여 특징 추출"""
    features_dict = {}
    model = model.to(device)
    model.eval()
    
    print("\n특징 추출 시작...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing"):
            try:
                # 데이터 디바이스로 이동
                x = batch['keypoints'].to(device)
                paths = batch['path']
                
                # backbone 특징 추출
                feat = model.extract_feat(x, stage='backbone')
                if isinstance(feat, tuple):
                    feat = feat[0]
                
                # 전역 평균 풀링
                features = F.adaptive_avg_pool3d(feat, (1, 1, 1)).squeeze()
                
                # 결과 저장
                for path, f in zip(paths, features.cpu().numpy()):
                    features_dict[path] = f
                    print(f"Extracted features shape for {Path(path).stem}: {f.shape}")
            
            except Exception as e:
                print(f"\nError processing batch: {e}")
                continue
    
    print(f"\n총 {len(features_dict)}개 시퀀스에서 특징 추출 완료")
    return features_dict

def save_features(features_dict, save_dir):
    """추출된 특징을 .pkl 파일로 저장"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 클래스별로 특징 저장
    for path, features in features_dict.items():
        pkl_path = save_dir / f"{Path(path).stem}_features.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump({
                'features': features,
                'source_path': path
            }, f)

def visualize_features(features_dict, save_dir, method='tsne'):
    """특징 벡터를 2D로 시각화"""
    if not features_dict:
        print("시각화할 특징이 없습니다.")
        return
    
    # 특징 벡터와 파일 이름 추출
    features = np.array(list(features_dict.values()))
    file_names = [Path(p).stem for p in features_dict.keys()]
    
    print(f"\n시각화 시작... ({method})")
    print(f"- 특징 데이터 형태: {features.shape}")
    
    # perplexity 조정
    n_samples = len(features)
    perplexity = min(30.0, n_samples - 1)  # 샘플 수에 따라 자동 조정
    
    # 차원 축소
    if method == 'tsne':
        reducer = TSNE(n_components=2, 
                      random_state=42,
                      perplexity=perplexity)
        embedding = reducer.fit_transform(features)
        title = f't-SNE Visualization (perplexity={perplexity})'
    else:
        reducer = PCA(n_components=2, random_state=42)
        embedding = reducer.fit_transform(features)
        explained_var = reducer.explained_variance_ratio_
        title = f'PCA Visualization\nExplained variance: {explained_var[0]:.2f}, {explained_var[1]:.2f}'
    
    # 시각화
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.6)
    
    # 레이블 추가
    step = max(1, len(embedding) // 20)  # 최대 20개 레이블만 표시
    for i, (x, y) in enumerate(embedding):
        if i % step == 0:
            plt.annotate(file_names[i], (x, y), fontsize=8)
    
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    
    # 저장
    save_path = Path(save_dir) / f'features_visualization_{method}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"시각화 결과가 {save_path}에 저장되었습니다.")

def main():
    # 1. 설정
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    batch_size = 16
    
    # 2. 모델 설정 및 로드
    from mmaction.apis import init_recognizer
    from mmengine.config import Config
    
    # 기본 설정을 딕셔너리로 정의
    cfg_dict = dict(
        model=dict(
            type='RecognizerGCN',
            backbone=dict(
                type='STGCN',
                graph_cfg=dict(
                    layout='coco',
                    mode='spatial',
                    max_hop=1
                ),
                in_channels=3,   # (x, y, confidence)
                base_channels=64,
                num_stages=10,
                inflate_stages=[5, 8],
                down_stages=[5, 8],
                num_person=1,    # 단일 인물
            ),
            cls_head=dict(
                type='GCNHead',
                num_classes=60,  # 원래 모델의 클래스 수로 맞춤
                in_channels=256,
                dropout=0.2
            )
        ),
        data_preprocessor=dict(
            type='ActionDataPreprocessor',
            mean=[0.5],
            std=[0.5],
            format_shape='NCHW'
        )
    )
    
    # Config 객체 생성
    config = Config(cfg_dict)
    
    # 체크포인트 경로
    checkpoint_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                 'models', 'ST_GCN_fet', 'ST_GCN_fet.pth')
    
    # 모델 초기화
    try:
        model = init_recognizer(config, checkpoint_file, device=device)
        print("\nST-GCN 모델 로드 성공")
        print(f"- 체크포인트: {checkpoint_file}")
        print(f"- 그래프 레이아웃: COCO (17 키포인트)")
    except Exception as e:
        print(f"모델 로드 실패: {str(e)}")
        print(f"체크포인트 파일 존재: {os.path.exists(checkpoint_file)}")
        return
    
    # 3. 데이터셋 설정
    class_name = 'lunge'  # 처리할 클래스 이름
    npz_dir = Path('data/normalized') / class_name  # 경로 수정
    
    if not npz_dir.exists():
        print(f"\n오류: NPZ 파일 디렉토리를 찾을 수 없습니다: {npz_dir}")
        return
        
    dataset = NpzSkeletonDataset(npz_dir)
    if len(dataset) == 0:
        print(f"\n오류: {npz_dir}에서 NPZ 파일을 찾을 수 없습니다.")
        return
        
    print(f"\n데이터셋 정보:")
    print(f"- NPZ 파일 경로: {npz_dir}")
    print(f"- 총 시퀀스 수: {len(dataset)}")
    
    dataloader = DataLoader(dataset, 
                          batch_size=batch_size, 
                          shuffle=False, 
                          num_workers=2)
    
    # 4. 특징 추출
    def extract_features(model, dataloader, device):
        features_dict = {}
        model.eval()
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="특징 추출"):
                try:
                    x = batch['keypoints'].to(device)
                    
                    # backbone 특징 추출 - 마지막 conv 레이어 이후
                    feat = model.extract_feat(x, stage='backbone')
                    if isinstance(feat, tuple):
                        feat = feat[0]
                    
                    # shape 출력으로 디버깅
                    print(f"\nInput shape: {x.shape}")
                    print(f"Feature shape: {feat.shape}")
                    
                    # 전역 평균 풀링
                    features = F.adaptive_avg_pool3d(feat, (1, 1, 1)).squeeze()
                    print(f"Pooled feature shape: {features.shape}")
                    
                    # 결과 저장
                    for path, feat in zip(batch['path'], features.cpu().numpy()):
                        features_dict[path] = feat
                        print(f"저장된 특징 shape ({Path(path).stem}): {feat.shape}")
                
                except Exception as e:
                    print(f"\n배치 처리 중 에러: {e}")
                    continue
        
        return features_dict
    
    # 5. 특징 추출 및 저장
    features_dict = extract_features(model, dataloader, device)
    
    # 6. 저장 및 시각화
    save_dir = Path('data') / class_name / 'features'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 특징 저장
    save_features(features_dict, save_dir)
    
    # 시각화
    visualize_features(features_dict, save_dir, method='tsne')
    visualize_features(features_dict, save_dir, method='pca')
    
    print(f"\n특징이 {save_dir}에 저장되었습니다.")

if __name__ == '__main__':
    main()