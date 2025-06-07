import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from run_train import train_class_name

def get_num_classes(class_name):
    """LABEL_MAP에서 클래스 수 읽어오기"""
    label_map_path = os.path.join('data', class_name, 'LABEL_MAP.json')
    try:
        with open(label_map_path, 'r') as f:
            label_map = json.load(f)
            # unlabeled(-1)를 제외한 클래스 수 계산
            return max(label_map.values()) + 1
    except FileNotFoundError:
        print(f"경고: {label_map_path}를 찾을 수 없습니다.")
        return None

def parse_log_file(log_file, class_name):
    # 클래스 수 가져오기
    num_classes = get_num_classes(class_name)
    if num_classes is None:
        raise ValueError(f"LABEL_MAP을 찾을 수 없습니다: {class_name}")
    
    metrics = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_acc': [],
        'confusion_matrices': []
    }
    
    current_epoch = None
    
    with open(log_file, 'r') as f:
        for line in f:
            # 에폭 시작 확인
            if 'Epoch(train)' in line:
                epoch_match = re.search(r'\[(\d+)\]', line)
                if epoch_match:
                    current_epoch = int(epoch_match.group(1))
                    
                # 훈련 메트릭 추출
                if ']  lr:' in line:
                    loss_match = re.search(r'loss: ([\d.]+)', line)
                    acc_match = re.search(r'top1_acc: ([\d.]+)', line)
                    if loss_match and acc_match:
                        metrics['epoch'].append(current_epoch)
                        metrics['train_loss'].append(float(loss_match.group(1)))
                        metrics['train_acc'].append(float(acc_match.group(1)))
            
            # 검증 정확도 추출
            if 'acc/top1:' in line:
                val_acc_match = re.search(r'acc/top1: ([\d.]+)', line)
                if val_acc_match:
                    metrics['val_acc'].append(float(val_acc_match.group(1)))
            
            # 혼동 행렬 추출
            if 'confusion_matrix/result:' in line:
                matrix_text = ''
                for _ in range(num_classes):  # 동적 크기 적용
                    matrix_line = next(f)
                    matrix_text += matrix_line
                # 문자열에서 숫자만 추출
                numbers = re.findall(r'\d+', matrix_text)
                matrix = np.array(numbers, dtype=int).reshape(num_classes, num_classes)
                metrics['confusion_matrices'].append(matrix)
    
    return metrics

def plot_training_curves(metrics, save_dir):
    plt.figure(figsize=(12, 6))
    
    # 손실과 정확도 그래프
    plt.subplot(1, 2, 1)
    plt.plot(metrics['epoch'], metrics['train_loss'], label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(metrics['epoch'], metrics['train_acc'], label='Train Accuracy')
    plt.plot(range(len(metrics['val_acc'])), metrics['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_curves.png')
    plt.close()

def plot_final_confusion_matrix(matrix, save_dir):
    # visualize_matrix.py의 함수 사용
    from visualize_matrix import visualize_confusion_matrix
    visualize_confusion_matrix(matrix, f'{save_dir}/final_confusion_matrix.png')

def find_latest_log(class_name):
    """클래스 폴더에서 가장 최근의 로그 파일 찾기"""
    base_dir = os.path.join('models', class_name)
    
    # 1. 클래스 폴더 확인
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"클래스 폴더를 찾을 수 없습니다: {base_dir}")
    
    # 2. 타임스탬프 폴더 찾기
    timestamp_dirs = []
    for d in os.listdir(base_dir):
        dir_path = os.path.join(base_dir, d)
        if os.path.isdir(dir_path) and d.startswith('202'):  # 2024, 2025 등으로 시작하는 폴더
            timestamp_dirs.append(dir_path)
    
    if not timestamp_dirs:
        raise FileNotFoundError(f"로그 파일을 찾을 수 없습니다: {base_dir}")
    
    # 3. 가장 최근 타임스탬프 폴더 선택
    latest_dir = max(timestamp_dirs)
    
    # 4. 로그 파일 찾기
    log_files = [f for f in os.listdir(latest_dir) if f.endswith('.log')]
    if not log_files:
        raise FileNotFoundError(f"로그 파일을 찾을 수 없습니다: {latest_dir}")
    
    # 5. 전체 경로 반환
    log_path = os.path.join(latest_dir, max(log_files))
    print(f"최신 로그 파일 발견: {log_path}")
    
    return log_path, latest_dir

if __name__ == "__main__":
    class_name = train_class_name  # 현재 처리 중인 클래스
    
    try:
        # 최신 로그 파일과 저장 디렉토리 자동 찾기
        log_file, save_dir = find_latest_log(class_name)
    except FileNotFoundError as e:
        print(f"오류: {e}")
        exit(1)
    
    # 결과 저장 경로가 없으면 생성
    os.makedirs(save_dir, exist_ok=True)
    
    # 로그 파싱
    metrics = parse_log_file(log_file, class_name)
    
    # 학습 곡선 그리기
    plot_training_curves(metrics, save_dir)
    
    # 마지막 혼동 행렬 시각화
    if metrics['confusion_matrices']:
        plot_final_confusion_matrix(metrics['confusion_matrices'][-1], save_dir)
    
    # 최종 성능 출력
    print("\n최종 성능 지표:")
    print(f"Train Accuracy: {metrics['train_acc'][-1]:.4f}")
    print(f"Validation Accuracy: {metrics['val_acc'][-1]:.4f}")