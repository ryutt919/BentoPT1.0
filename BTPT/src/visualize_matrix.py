import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

def get_label_names():
    label_map_path = r"C:\Users\kimt9\Desktop\RyuTTA\2025_3_1\ComputerVision\TermP\BTPT\data\pull_ups\LABEL_MAP.json"
    try:
        with open(label_map_path, 'r', encoding='utf-8') as f:
            label_map = json.load(f)
            # 값을 기준으로 정렬된 레이블 이름 리스트 생성
            sorted_labels = sorted([(v, k) for k, v in label_map.items() if v >= 0])
            return [label[1] for label in sorted_labels]
    except Exception as e:
        print(f"LABEL_MAP 로드 실패: {str(e)}")
        return [f'Class {i}' for i in range(9)]

def visualize_confusion_matrix(matrix, save_path):
    # 한글 폰트 설정
    plt.rcParams['font.family'] = 'Malgun Gothic'
    
    class_names = get_label_names()
    
    # 플롯 크기 설정
    plt.figure(figsize=(20, 16))
    
    # 1. Raw Counts 히트맵
    plt.subplot(2, 1, 1)
    sns.heatmap(matrix, 
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('혼동 행렬 (원본 카운트)', fontsize=15, pad=20)
    plt.xlabel('예측 레이블')
    plt.ylabel('실제 레이블')
    
    # x축 레이블 회전
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # 2. 정규화된 히트맵 (퍼센트)
    plt.subplot(2, 1, 2)
    row_sums = matrix.sum(axis=1)
    row_sums[row_sums == 0] = 1  # 0으로 나누기 방지
    matrix_norm = matrix.astype('float') / row_sums[:, np.newaxis]
    
    sns.heatmap(matrix_norm,
                annot=True,
                fmt='.1%',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('혼동 행렬 (정규화)', fontsize=15, pad=20)
    plt.xlabel('예측 레이블')
    plt.ylabel('실제 레이블')
    
    # x축 레이블 회전
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # 레이아웃 조정 및 저장
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 성능 분석 결과 출력
    print("\n자세별 성능 분석:")
    print("-" * 50)
    total_samples = 0
    correct_samples = 0
    
    for i, class_name in enumerate(class_names):
        true_pos = matrix[i, i]
        total = matrix[i].sum()
        if total > 0:
            accuracy = (true_pos / total) * 100
            total_samples += total
            correct_samples += true_pos
            print(f"{class_name:>20}: {accuracy:.1f}% ({true_pos}/{total})")
    
    print("-" * 50)
    overall_accuracy = (correct_samples / total_samples * 100) if total_samples > 0 else 0
    print(f"전체 정확도: {overall_accuracy:.1f}% ({correct_samples}/{total_samples})")