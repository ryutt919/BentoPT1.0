import numpy as np
from scipy.ndimage import median_filter
import os
from tqdm import tqdm

# 1) NPZ 파일 로드 및 내부 확인
npz_path = r"C:\Users\kimt9\Desktop\RyuTTA\2025_3_1\ComputerVision\TermP\mmaction2\data\kinetics400\keypoints\bench pressing\bench_press (157).npz"
dst_npz = r"C:\Users\kimt9\Desktop\RyuTTA\2025_3_1\ComputerVision\TermP\mmaction2\data\kinetics400\keypoints\bench pressing\bench_press (157)2.npz"

data = np.load(npz_path, allow_pickle=True)
print("Archive contains:", data.files)

# keypoints 배열 가져오기 (shape = (T, K, 3))
kp = data["keypoints"]
print("Archive shape:", kp.shape)  # e.g., (26, 13, 3)

# 2) Median 필터 함수 정의
def fill_nan_with_neighbors(arr):
    """선/후행 유효값으로 NaN을 보간"""
    n = len(arr)
    idx = np.arange(n)
    mask = np.isnan(arr)
    if mask.all():
        return np.zeros_like(arr)
    arr[mask] = np.interp(idx[mask], idx[~mask], arr[~mask])
    return arr


def median_smooth_keypoints(kp_array, kernel_size=5):
    """
    kp_array: np.ndarray, shape = (T, K, 3)
      - T: 시퀀스 길이
      - K: 키포인트 개수
      - 마지막 축 [x, y, confidence]
    kernel_size: 홀수(3, 5, 7 등)
    반환: same shape, x/y 좌표만 median 필터 적용된 배열
    """
    T, K, _ = kp_array.shape
    smoothed = kp_array.copy()

    for k in range(K):
        xs = kp_array[:, k, 0].astype(float)
        ys = kp_array[:, k, 1].astype(float)
        conf = kp_array[:, k, 2]

        xs[conf <= 0] = np.nan
        ys[conf <= 0] = np.nan

        xs = fill_nan_with_neighbors(xs)
        ys = fill_nan_with_neighbors(ys)

        xs_med = median_filter(xs, size=kernel_size, mode="nearest")
        ys_med = median_filter(ys, size=kernel_size, mode="nearest")

        smoothed[:, k, 0] = xs_med
        smoothed[:, k, 1] = ys_med

    return smoothed

# 3) Median 필터 적용 (kernel_size=5 기준)
kp_smoothed = median_smooth_keypoints(kp, kernel_size=7)

# 4) 나머지 메타데이터 그대로 유지하면서 새 npz로 저장
np.savez_compressed(
    dst_npz,
    frame_numbers=data["frame_numbers"],
    timestamps=data["timestamps"],
    person_ids=data["person_ids"],
    keypoints=kp_smoothed,
    keypoint_names=data["keypoint_names"],
    video_info=data["video_info"]  # allow_pickle=True로 불러왔으니 이대로 저장 가능
)

print(f"Smoothed keypoints saved to: {dst_npz}")

def process_all_keypoints():
    # 기본 경로 설정
    base_dir = r"C:\Users\kimt9\Desktop\RyuTTA\2025_3_1\ComputerVision\TermP\mmaction2\data\kinetics400"
    src_dir = os.path.join(base_dir, "keypoints")
    dst_dir = os.path.join(base_dir, "smoothed")
    
    # 결과 저장할 디렉토리 생성
    os.makedirs(dst_dir, exist_ok=True)
    
    # 모든 클래스 폴더 가져오기
    class_dirs = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]
    print(f"총 {len(class_dirs)}개 클래스 처리 예정")
    
    # 전체 파일 수 계산
    total_files = sum(len([f for f in os.listdir(os.path.join(src_dir, d)) if f.endswith('.npz')]) for d in class_dirs)
    processed_files = 0
    
    # 각 클래스별로 처리
    for class_name in tqdm(class_dirs, desc="클래스 처리 중"):
        # 클래스별 입/출력 디렉토리 설정
        src_class_dir = os.path.join(src_dir, class_name)
        dst_class_dir = os.path.join(dst_dir, class_name)
        os.makedirs(dst_class_dir, exist_ok=True)
        
        # 클래스 내의 모든 npz 파일 처리
        npz_files = [f for f in os.listdir(src_class_dir) if f.endswith('.npz')]
        
        for npz_file in tqdm(npz_files, desc=f"{class_name} 처리 중", leave=False):
            src_path = os.path.join(src_class_dir, npz_file)
            dst_path = os.path.join(dst_class_dir, npz_file)
            
            # 이미 처리된 파일은 건너뛰기
            if os.path.exists(dst_path):
                processed_files += 1
                continue
                
            try:
                # NPZ 파일 로드
                data = np.load(src_path, allow_pickle=True)
                
                # Median 필터 적용
                kp = data["keypoints"]
                kp_smoothed = median_smooth_keypoints(kp, kernel_size=5)
                
                # 결과 저장
                np.savez_compressed(
                    dst_path,
                    frame_numbers=data["frame_numbers"],
                    timestamps=data["timestamps"],
                    person_ids=data["person_ids"],
                    keypoints=kp_smoothed,
                    keypoint_names=data["keypoint_names"],
                    video_info=data["video_info"]
                )
                processed_files += 1
                
            except Exception as e:
                print(f"\nError processing {src_path}: {str(e)}")
                continue
            
            # 진행 상황 출력
            print(f"\r전체 진행률: {processed_files}/{total_files} ({processed_files/total_files*100:.1f}%)", end="")

if __name__ == "__main__":
    process_all_keypoints()
    print("\n모든 처리가 완료되었습니다.")
