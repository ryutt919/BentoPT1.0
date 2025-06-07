import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from mmengine.config import Config
from mmaction.apis import init_recognizer
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from run_train import train_class_name

class UnlabeledDataset(Dataset):
    def __init__(self, pkl_path):
        """
        unlabeled.pkl 파일에서 라벨이 없는 데이터만 로드하는 데이터셋
        """
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        self.unlabeled_frames = data['split']['unlabeled']
        self.annotations = {
            item['frame_dir']: item 
            for item in data['annotations']
            if item['frame_dir'] in self.unlabeled_frames
        }
        
        # 첫 번째 아이템의 shape 출력
        first_item = next(iter(self.annotations.values()))
        print(f"Raw keypoint shape: {first_item['keypoint'].shape}")
        
    def __len__(self):
        return len(self.unlabeled_frames)
    
    def __getitem__(self, idx):
        frame_dir = self.unlabeled_frames[idx]
        item = self.annotations[frame_dir]
        
        # 키포인트 데이터 가져오기
        keypoints = item['keypoint']  # (M, T, V, 2)
        
        # z축 좌표를 0으로 추가
        M, T, V, C = keypoints.shape
        keypoints_3d = np.zeros((1, M, T, V, 3), dtype=np.float32)  # (N=1, M, T, V, C=3)
        keypoints_3d[0, ..., :2] = keypoints  # x, y 좌표 복사
        # z 좌표는 0으로 초기화되어 있음
        
        print(f"Dataset keypoint shape: {keypoints_3d.shape}")
        
        return {
            'keypoint': torch.FloatTensor(keypoints_3d),  # (N=1, M, T, V, C)
            'frame_dir': frame_dir
        }

def get_predictions(model, dataloader, confidence_threshold=0.8):
    """Get predictions for unlabeled data with confidence scores"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating predictions"):
            # DataLoader가 첫 번째 차원(N=1)들을 배치로 합침
            keypoints = batch['keypoint'].cuda()
            
            # 1. backbone을 통과하여 특징 추출
            feat = model.extract_feat(keypoints)[0]
            
            # 2. classification head를 통과하여 클래스 점수 얻기
            cls_score = model.cls_head(feat)
            
            # 소프트맥스로 확률 변환
            probs = torch.softmax(cls_score, dim=1)
            confidence, predicted = torch.max(probs, dim=1)
            
            # Filter predictions based on confidence threshold
            mask = confidence >= confidence_threshold
            high_conf_indices = torch.where(mask)[0]
            
            for idx in high_conf_indices:
                predictions.append({
                    'frame_dir': batch['frame_dir'][idx],
                    'label': predicted[idx].item(),
                    'confidence': confidence[idx].item()
                })
            
            # Print progress for each batch
            print(f"Current batch: found {len(high_conf_indices)} samples with confidence >= {confidence_threshold}")
    
    return predictions

def update_dataset_with_pseudo_labels(train_pkl_path, predictions, label_encoder):
    """Update training dataset with pseudo-labeled data"""
    # Load existing training dataset
    with open(train_pkl_path, 'rb') as f:
        train_dataset = pickle.load(f)
    
    # Add pseudo-labeled data to training dataset
    for pred in predictions:
        # 라벨은 이미 숫자 형태이므로 변환하지 않음
        label = pred['label']  # 이미 0-12 사이의 정수
        
        # Add frame_dir to training split
        if pred['frame_dir'] not in train_dataset['split']['xsub_train']:
            train_dataset['split']['xsub_train'].append(pred['frame_dir'])
        
        # Update annotation with pseudo-label
        for ann in train_dataset['annotations']:
            if ann['frame_dir'] == pred['frame_dir']:
                ann['label'] = int(label)  # 정수로 저장
                ann['pseudo_labeled'] = True
                ann['confidence'] = pred['confidence']
                break
    
    # Save updated dataset
    with open(train_pkl_path, 'wb') as f:
        pickle.dump(train_dataset, f)

def main():
    # Load config and setup
    cfg = Config.fromfile('configs/custom_config.py')  # 사용자 정의 config 파일 경로
    class_name = train_class_name  # lunge로 수정
    
    # lunge 동작의 레이블로 초기화
    label_encoder = LabelEncoder()
    all_labels = [
        "outlier",
        "good",
        "shoulder_raise",
        "grip_narrow",
        "torso_shaking",
        "spine_imbalance",
        "shoulder_asymmetry",
        "max_contraction_needed",
        "max_relaxation_needed"
    ]
    label_encoder.fit(all_labels)
    
    # Initialize model
    model = init_recognizer(cfg, device='cuda')
    
    # Load checkpoint if exists
    checkpoint_path = r'C:\Users\kimt9\Desktop\RyuTTA\2025_3_1\ComputerVision\TermP\BTPT\models\pull_ups\best_acc_top1_epoch_42.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print("No checkpoint found. Please train the model first.")
        return
    
    # Load unlabeled dataset - 경로도 lunge로 수정
    unlabeled_dataset = UnlabeledDataset(os.path.join('data', class_name, 'pkl', 'unlabeled.pkl'))
    print(f"Loaded {len(unlabeled_dataset)} unlabeled samples")
    
    # Create dataloader for unlabeled data
    unlabeled_dataloader = DataLoader(
        unlabeled_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4
    )
    
    # Generate predictions
    predictions = get_predictions(model, unlabeled_dataloader)
    print(f"Generated predictions for {len(predictions)} samples with confidence >= 0.8")
    
    # Update training dataset with pseudo-labels - 경로도 lunge로 수정
    update_dataset_with_pseudo_labels(
        os.path.join('data', class_name, 'pkl', 'train.pkl'),
        predictions,
        label_encoder
    )
    
    print(f"Added {len(predictions)} pseudo-labeled samples to training dataset")

if __name__ == '__main__':
    main()