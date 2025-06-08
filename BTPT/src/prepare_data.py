import os
import numpy as np
import json
import pickle
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from run_train import train_class_name

def load_label_map():
    class_name = train_class_name
    label_map_path = os.path.join('data', class_name, 'LABEL_MAP.json')
    try:
        with open(label_map_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"\n경고: {label_map_path}를 찾을 수 없습니다.")
        return None

def get_sequence_windows(keypoints, frame_numbers, labels_df, window_size=8, stride=4):
    """슬라이딩 윈도우 방식으로 시퀀스 생성"""
    T = len(frame_numbers)
    sequences = []
    
    # 시퀀스가 너무 짧으면 건너뛰기
    if T < window_size:
        print(f"Warning: Sequence length ({T}) is shorter than window size ({window_size})")
        return sequences
    
    for start_idx in range(0, T - window_size + 1, stride):
        end_idx = start_idx + window_size
        
        # 현재 윈도우의 키포인트
        window_keypoints = keypoints[start_idx:end_idx]
        
        # 현재 윈도우의 프레임 번호
        window_frames = frame_numbers[start_idx:end_idx]
        
        # 현재 윈도우에 해당하는 라벨 가져오기
        if labels_df is not None:
            window_labels = labels_df[labels_df['frame'].isin(window_frames)]
            if not window_labels.empty:
                # 각 라벨의 등장 횟수 계산
                label_counts = window_labels['labels'].value_counts()
                # 가장 많이 등장한 라벨 선택
                most_common_label = label_counts.index[0]
                label = LABEL_MAP.get(most_common_label, -1)
            else:
                label = -1
        else:
            label = -1
        
        sequences.append({
            'keypoints': window_keypoints,
            'frames': window_frames,
            'label': label
        })
    
    return sequences

def process_npz_file(npz_path, csv_path=None):
    """NPZ 파일을 처리하여 MMACTION2 형식의 데이터로 변환"""
    data = np.load(npz_path, allow_pickle=True)
    keypoints = data['keypoints']  # (T, V, 3)
    frame_numbers = data['frame_numbers']
    
    print(f"\nProcessing {Path(npz_path).stem}:")
    print(f"  - Total frames: {len(frame_numbers)}")
    
    # CSV에서 라벨 정보 읽기
    labels_df = None
    if csv_path and os.path.exists(csv_path):
        labels_df = pd.read_csv(csv_path)
        print(f"  - Found label file with {len(labels_df)} labeled frames")
    else:
        print(f"  - No label file found")
    
    # COCO 포맷에 맞게 키포인트 재구성
    # COCO 순서: [nose, left_eye, right_eye, left_ear, right_ear, 
    #            left_shoulder, right_shoulder, left_elbow, right_elbow,
    #            left_wrist, right_wrist, left_hip, right_hip,
    #            left_knee, right_knee, left_ankle, right_ankle]
    
    # YOLOv8 키포인트 매핑
    YOLO_TO_COCO = {
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
    
    T = len(frame_numbers)
    coco_kpts = np.zeros((T, 17, 3))  # 17개의 COCO 키포인트
    
    # 있는 키포인트 복사
    for yolo_idx, coco_idx in YOLO_TO_COCO.items():
        if yolo_idx < keypoints.shape[1]:
            coco_kpts[:, coco_idx] = keypoints[:, yolo_idx]
    
    # 없는 키포인트 추정 (눈, 귀)
    # 코와 어깨의 중간점을 사용
    nose_idx = 0
    left_shoulder_idx = 5
    right_shoulder_idx = 6
    
    # 눈과 귀의 위치를 코와 어깨 사이의 3:7 지점으로 추정
    for t in range(T):
        if coco_kpts[t, nose_idx, 2] > 0 and coco_kpts[t, left_shoulder_idx, 2] > 0:
            # 왼쪽 눈과 귀
            coco_kpts[t, 1] = 0.7 * coco_kpts[t, nose_idx] + 0.3 * coco_kpts[t, left_shoulder_idx]  # left_eye
            coco_kpts[t, 3] = 0.3 * coco_kpts[t, nose_idx] + 0.7 * coco_kpts[t, left_shoulder_idx]  # left_ear
            coco_kpts[t, [1, 3], 2] = min(coco_kpts[t, nose_idx, 2], coco_kpts[t, left_shoulder_idx, 2])
        
        if coco_kpts[t, nose_idx, 2] > 0 and coco_kpts[t, right_shoulder_idx, 2] > 0:
            # 오른쪽 눈과 귀
            coco_kpts[t, 2] = 0.7 * coco_kpts[t, nose_idx] + 0.3 * coco_kpts[t, right_shoulder_idx]  # right_eye
            coco_kpts[t, 4] = 0.3 * coco_kpts[t, nose_idx] + 0.7 * coco_kpts[t, right_shoulder_idx]  # right_ear
            coco_kpts[t, [2, 4], 2] = min(coco_kpts[t, nose_idx, 2], coco_kpts[t, right_shoulder_idx, 2])
    
    # 키포인트와 confidence score 분리
    kp = coco_kpts[:, :, :2]  # (T, V, 2)
    kpscore = coco_kpts[:, :, 2]  # (T, V)
    
    # 슬라이딩 윈도우로 시퀀스 생성
    sequences = get_sequence_windows(kp, frame_numbers, labels_df)
    print(f"  - Generated {len(sequences)} sequences")
    
    # 각 시퀀스를 MMACTION2 형식으로 변환
    results = []
    for i, seq in enumerate(sequences):
        # 키포인트 데이터 형태 변환: (T, V, 2) -> (M, T, V, 2) where M=1 (single person)
        kp_seq = np.expand_dims(seq['keypoints'], axis=0)  # (1, T, V, 2)
        kpscore_seq = np.expand_dims(kpscore[i:i+len(seq['frames'])], axis=0)  # (1, T, V)
        
        results.append({
            'frame_dir': f"{Path(npz_path).stem}_seq{i}",
            'total_frames': len(seq['frames']),
            'label': seq['label'],
            'keypoint': kp_seq,  # (M, T, V, 2)
            'keypoint_score': kpscore_seq,  # (M, T, V)
            'num_person': 1
        })
    
    return results

def prepare_dataset():
    # 기본 경로 설정
    class_name = train_class_name
    base_dir = Path('data')
    npz_dir = base_dir / 'normalized' / class_name
    csv_dir = base_dir / 'labels' / class_name
    
    # 클래스별 저장 경로 설정
    output_dir = Path('data') / class_name / 'pkl'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n저장 경로: {output_dir}")
    
    # 데이터 리스트 생성
    annotations = []
    npz_files = list(npz_dir.glob('*.npz'))
    
    print(f"\nFound {len(npz_files)} NPZ files")
    
    for npz_path in tqdm(npz_files, desc="Processing files"):
        csv_path = csv_dir / f"{npz_path.stem}.csv"
        try:
            items = process_npz_file(npz_path, csv_path)
            annotations.extend(items)
        except Exception as e:
            print(f"Error processing {npz_path}: {e}")
    
    # 라벨이 있는 데이터와 없는 데이터 분리
    labeled_data = [d for d in annotations if d['label'] != -1]
    unlabeled_data = [d for d in annotations if d['label'] == -1]
    
    print(f"\n총 시퀀스 수: {len(annotations)}")
    print(f"  - 라벨 있는 시퀀스: {len(labeled_data)}")
    print(f"  - 라벨 없는 시퀀스: {len(unlabeled_data)}")
    
    # 데이터 분할 처리
    if len(labeled_data) > 0:
        # 라벨이 있는 데이터를 학습/검증 세트로 분할
        train_data, val_data = train_test_split(labeled_data, test_size=0.2, random_state=42)
    else:
        print("\n경고: 라벨이 있는 데이터가 없습니다!")
        print("모든 데이터를 라벨 없는 데이터로 처리합니다.")
        train_data = []
        val_data = []
    
    # MMACTION2 형식으로 데이터 저장
    # 1. 학습 데이터
    train_dataset = {
        'split': {
            'xsub_train': [item['frame_dir'] for item in train_data]
        },
        'annotations': annotations  # 모든 데이터의 annotation 포함
    }
    
    # 2. 검증 데이터
    val_dataset = {
        'split': {
            'xsub_val': [item['frame_dir'] for item in val_data]
        },
        'annotations': annotations  # 모든 데이터의 annotation 포함
    }
    
    # 3. 라벨 없는 데이터 (모든 데이터가 라벨이 없는 경우도 포함)
    unlabeled_dataset = {
        'split': {
            'unlabeled': [item['frame_dir'] for item in (unlabeled_data if len(labeled_data) > 0 else annotations)]
        },
        'annotations': annotations
    }
    
    # 저장 디렉토리 생성
    output_dir = Path('data') / class_name / 'pkl'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Pickle 파일로 저장
    with open(output_dir / 'train.pkl', 'wb') as f:
        pickle.dump(train_dataset, f)
    
    with open(output_dir / 'val.pkl', 'wb') as f:
        pickle.dump(val_dataset, f)
    
    with open(output_dir / 'unlabeled.pkl', 'wb') as f:
        pickle.dump(unlabeled_dataset, f)
    
    print(f"\n데이터셋 통계:")
    print(f"저장 위치: {output_dir}")
    print(f"훈련 샘플: {len(train_data)}")
    print(f"검증 샘플: {len(val_data)}")
    print(f"레이블 없는 샘플: {len(unlabeled_data)}")

    # 라벨 분포 출력
    if len(labeled_data) > 0:
        labels = [d['label'] for d in labeled_data]
        unique_labels, counts = np.unique(labels, return_counts=True)
        print("\n라벨 분포:")
        for label, count in zip(unique_labels, counts):
            print(f"  - 라벨 {label}: {count}개")

if __name__ == '__main__':
    global LABEL_MAP
    LABEL_MAP = load_label_map()
    prepare_dataset()