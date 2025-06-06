import os
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def find_latest_log():
    """가장 최근의 로그 파일 찾기"""
    base_dir = Path("TermP/mmaction2/work_dirs/custom_stgcnpp")
    
    # 날짜_시간 형식의 디렉토리 찾기
    exp_dirs = []
    for d in base_dir.iterdir():
        if d.is_dir() and re.match(r"\d{8}_\d{6}", d.name):
            exp_dirs.append(d)
    
    if not exp_dirs:
        raise FileNotFoundError("로그 디렉토리를 찾을 수 없습니다.")
    
    # 가장 최근 디렉토리 찾기
    latest_dir = max(exp_dirs, key=lambda x: datetime.strptime(x.name, "%Y%m%d_%H%M%S"))
    log_file = latest_dir / f"{latest_dir.name}.log"
    
    if not log_file.exists():
        raise FileNotFoundError(f"로그 파일을 찾을 수 없습니다: {log_file}")
    
    return log_file

def parse_log_file(log_file):
    """로그 파일 파싱"""
    train_metrics = {
        'epoch': [],
        'loss': [],
        'top1_acc': [],
        'top5_acc': []
    }
    
    val_metrics = {
        'epoch': [],
        'top1_acc': [],
        'top5_acc': [],
        'mean1_acc': []
    }
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            # 학습 메트릭 파싱
            if "Epoch(train)" in line:
                match = re.search(r"Epoch\(train\)\s+\[(\d+)].*loss: ([\d.]+).*top1_acc: ([\d.]+).*top5_acc: ([\d.]+)", line)
                if match:
                    epoch, loss, top1, top5 = match.groups()
                    train_metrics['epoch'].append(int(epoch))
                    train_metrics['loss'].append(float(loss))
                    train_metrics['top1_acc'].append(float(top1))
                    train_metrics['top5_acc'].append(float(top5))
            
            # 검증 메트릭 파싱
            if "Epoch(val)" in line:
                match = re.search(r"Epoch\(val\).*acc/top1: ([\d.]+).*acc/top5: ([\d.]+).*acc/mean1: ([\d.]+)", line)
                if match:
                    top1, top5, mean1 = match.groups()
                    val_metrics['epoch'].append(len(val_metrics['epoch']) + 1)
                    val_metrics['top1_acc'].append(float(top1))
                    val_metrics['top5_acc'].append(float(top5))
                    val_metrics['mean1_acc'].append(float(mean1))
    
    return pd.DataFrame(train_metrics), pd.DataFrame(val_metrics)

def plot_metrics(train_df, val_df, save_dir):
    """메트릭 시각화"""
    plt.style.use('seaborn')
    
    # Loss 그래프
    plt.figure(figsize=(12, 6))
    plt.plot(train_df['epoch'], train_df['loss'], label='Training Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss.png'))
    plt.close()
    
    # Accuracy 그래프
    plt.figure(figsize=(12, 6))
    plt.plot(train_df['epoch'], train_df['top1_acc'], label='Train Top-1 Acc')
    plt.plot(val_df['epoch'], val_df['top1_acc'], label='Val Top-1 Acc')
    plt.plot(train_df['epoch'], train_df['top5_acc'], label='Train Top-5 Acc')
    plt.plot(val_df['epoch'], val_df['top5_acc'], label='Val Top-5 Acc')
    plt.title('Training and Validation Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'accuracy.png'))
    plt.close()

def print_summary(train_df, val_df):
    """학습 결과 요약"""
    print("\n=== 학습 결과 요약 ===")
    print(f"총 에포크: {len(val_df)}")
    print("\n최종 성능:")
    print(f"  Training Loss: {train_df['loss'].iloc[-1]:.4f}")
    print(f"  Training Top-1 Accuracy: {train_df['top1_acc'].iloc[-1]:.4f}")
    print(f"  Validation Top-1 Accuracy: {val_df['top1_acc'].iloc[-1]:.4f}")
    print("\n최고 성능:")
    print(f"  Best Training Top-1 Accuracy: {train_df['top1_acc'].max():.4f}")
    print(f"  Best Validation Top-1 Accuracy: {val_df['top1_acc'].max():.4f}")
    print(f"  Best Validation Top-5 Accuracy: {val_df['top5_acc'].max():.4f}")

def main():
    try:
        # 최신 로그 파일 찾기
        log_file = find_latest_log()
        print(f"분석할 로그 파일: {log_file}")
        
        # 로그 파싱
        train_df, val_df = parse_log_file(log_file)
        
        # 그래프 저장 디렉토리 설정
        save_dir = log_file.parent
        
        # 메트릭 시각화
        plot_metrics(train_df, val_df, save_dir)
        print(f"\n그래프가 저장된 위치: {save_dir}")
        
        # 결과 요약 출력
        print_summary(train_df, val_df)
        
    except Exception as e:
        print(f"에러 발생: {str(e)}")

if __name__ == '__main__':
    main() 