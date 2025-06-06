import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from mmengine.config import Config
from mmengine.runner import Runner
from mmaction.apis import init_recognizer
from sklearn.preprocessing import LabelEncoder
from torch.utils.tensorboard import SummaryWriter

# TensorBoard writer
writer = SummaryWriter('runs/stgcn_experiment')

# Dataset class
class MultiCSVPoseDataset(Dataset):
    def __init__(self, base_dir, class_name, label_encoder, transform=None):
        self.samples = []
        self.label_encoder = label_encoder
        self.transform = transform
        self.window_size = 100  # 100프레임으로 변경

        npz_dir = os.path.join(base_dir, 'normalized', class_name)
        labels_dir = os.path.join(base_dir, 'labels', class_name)

        npz_files = [f for f in os.listdir(npz_dir) if f.endswith('.npz')]

        for npz_file in npz_files:
            npz_path = os.path.join(npz_dir, npz_file)
            csv_path = os.path.join(labels_dir, npz_file.replace('.npz', '.csv'))            
            data = np.load(npz_path, allow_pickle=True)
            keypoints = data['keypoints']  # (T, V, C)
            frame_numbers = data['frame_numbers']
            
            # CSV 파일에서 라벨 정보 읽기
            frame_labels = {}
            if os.path.exists(csv_path):
                try:
                    with open(csv_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        if len(lines) > 1:  # 헤더가 아닌 내용이 있는 경우
                            for line in lines[1:]:  # 헤더 제외
                                try:
                                    fields = line.strip().split(',')
                                    if len(fields) >= 3:
                                        frame = int(fields[0])
                                        labels = fields[2]
                                        frame_labels[frame] = labels.split(';')[0]  # 첫 번째 라벨만 사용
                                except Exception as e:
                                    print(f"Warning: Skipping line in {csv_path}: {e}")
                except Exception as e:
                    print(f"Warning: Error processing file {csv_path}: {e}")

            # 시퀀스를 windows로 나누기 (100프레임씩)
            stride = 50  # 절반씩 겹치도록
            
            for start_idx in range(0, len(frame_numbers), stride):
                end_idx = start_idx + self.window_size
                if end_idx > len(frame_numbers):
                    # 마지막 윈도우는 뒤에서부터 window_size만큼
                    if len(frame_numbers) >= self.window_size:
                        start_idx = len(frame_numbers) - self.window_size
                        end_idx = len(frame_numbers)
                    else:
                        # 영상이 window_size보다 짧은 경우
                        # 마지막 프레임을 반복하여 window_size만큼 채움
                        window_keypoints = np.pad(
                            keypoints,
                            ((0, self.window_size - len(keypoints)), (0, 0), (0, 0)),
                            mode='edge'
                        )
                        window_frames = frame_numbers
                        start_idx = 0
                        end_idx = self.window_size
                else:
                    window_keypoints = keypoints[start_idx:end_idx]
                    window_frames = frame_numbers[start_idx:end_idx]

                # 윈도우 내의 라벨 확인
                window_labels = []
                for frame in window_frames:
                    if frame in frame_labels:
                        window_labels.append(frame_labels[frame])

                if window_labels:
                    # 라벨이 있는 경우, 가장 많이 등장한 라벨 사용
                    from collections import Counter
                    most_common_label = Counter(window_labels).most_common(1)[0][0]
                    self.samples.append((window_keypoints, most_common_label))
                else:
                    # 라벨이 없는 경우, unlabeled로 처리 (준지도 학습용)
                    self.samples.append((window_keypoints, "unlabeled"))

                # 마지막 윈도우였다면 반복 종료
                if end_idx >= len(frame_numbers):
                    break

        if not self.samples:
            raise RuntimeError(f"No valid samples found in {base_dir}")

        self.X = [s[0] for s in self.samples]
        self.y = label_encoder.transform([s[1] for s in self.samples])

    def __getitem__(self, idx):
        keypoints = self.X[idx]  # (T, V, C)
        label = self.y[idx]
        
        # 데이터 포맷 변환 (T, V, C) -> (C, T, V)
        keypoints = np.transpose(keypoints, (2, 0, 1))
        
        # 텐서로 변환
        keypoints = torch.FloatTensor(keypoints)
        label = torch.LongTensor([label])[0]
        
        # 정규화 (이미 수행되었다면 생략 가능)
        if self.transform:
            keypoints = self.transform(keypoints)
            
        return {
            'keypoint': keypoints,
            'label': label
        }
        
    def __len__(self):
        return len(self.X)

class TensorBoardCallback:
    def __init__(self, writer):
        self.writer = writer
        self.step = 0
    
    def after_train_iter(self, runner):
        # 학습 손실 기록
        loss = runner.outputs['loss'].item()
        self.writer.add_scalar('Loss/train', loss, self.step)
        self.step += 1
    
    def after_val_epoch(self, runner):
        # 검증 정확도 기록
        accuracy = runner.metrics['accuracy']
        self.writer.add_scalar('Accuracy/val', accuracy, runner.epoch)

def main():
    # 설정 파일 로드
    cfg = Config.fromfile('mmaction2/configs/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py')
      # 데이터셋 준비
    label_encoder = LabelEncoder()
    label_encoder.fit(['unlabeled', 'outlier', 'good', 'elbow_not_vertical', 'hand_position_bad',
                      'elbow_flare', 'wrist_bent', 'shoulder_raise', 'grip_unstable',
                      'spine_imbalance', 'shoulder_asymmetry', 'barbell_unbalanced',
                      'max_contraction_needed', 'max_relaxation_needed'])
    
    # 전체 데이터셋 생성
    dataset = MultiCSVPoseDataset(
        base_dir='mmaction2/data/kinetics400',
        class_name='bench_pressing',
        label_encoder=label_encoder
    )
    
    # 데이터셋 분할 (80% 학습, 20% 검증)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size]
    )
    
    # 데이터 로더 설정
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.train_dataloader.batch_size,
        shuffle=True,
        num_workers=cfg.train_dataloader.num_workers
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.val_dataloader.batch_size,
        shuffle=False,
        num_workers=cfg.val_dataloader.num_workers
    )
    
    # 모델 초기화
    model = init_recognizer(cfg, device='cuda')
    
    # Runner 설정
    runner = Runner(
        model=model,
        work_dir='./work_dir/stgcn',
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        train_cfg=cfg.train_cfg,
        val_cfg=cfg.val_cfg,
        optim_wrapper=cfg.optim_wrapper,
        param_scheduler=cfg.param_scheduler,
        default_hooks=cfg.default_hooks,
        default_scope='mmaction'
    )
    
    # TensorBoard 콜백 추가
    tensorboard_callback = TensorBoardCallback(writer)
    runner.register_hook(tensorboard_callback)
    
    # 학습 시작
    runner.train()
    
    # TensorBoard writer 종료
    writer.close()

if __name__ == '__main__':
    main()
