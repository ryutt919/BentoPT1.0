import numpy as np
from scipy.ndimage import median_filter, gaussian_filter
import os
from tqdm import tqdm

def fill_nan_with_neighbors(arr):
    """선/후행 유효값으로 NaN을 보간"""
    n = len(arr)
    idx = np.arange(n)
    mask = np.isnan(arr)
    if mask.all():
        return np.zeros_like(arr)
    arr[mask] = np.interp(idx[mask], idx[~mask], arr[~mask])
    return arr

def smooth_keypoints(kp_array, median_kernel=5, gaussian_sigma=1.5):

    """
    kp_array: np.ndarray, shape = (T, K, 3)
      - T: 시퀀스 길이
      - K: 키포인트 개수
      - 마지막 축 [x, y, confidence]
    median_kernel: 홀수(3, 5, 7 등) - 메디안 필터 크기
    gaussian_sigma: 가우시안 필터 표준편차
    반환: same shape, x/y 좌표에 메디안→가우시안 필터가 적용된 배열
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

        # 메디안 필터로 이상치 제거
        xs_med = median_filter(xs, size=median_kernel, mode="nearest")
        ys_med = median_filter(ys, size=median_kernel, mode="nearest")

        # 가우시안 필터로 부드럽게 스무딩
        xs_smooth = gaussian_filter(xs_med, sigma=gaussian_sigma, mode="nearest")
        ys_smooth = gaussian_filter(ys_med, sigma=gaussian_sigma, mode="nearest")

        smoothed[:, k, 0] = xs_smooth
        smoothed[:, k, 1] = ys_smooth

    return smoothed

def process_all_keypoints():
    # 기본 경로 설정
    base_dir = r"C:\Users\kimt9\Desktop\RyuTTA\2025_3_1\ComputerVision\TermP\mmaction2\data\kinetics400"
    src_dir = os.path.join(base_dir, "keypoints")
    dst_dir = os.path.join(base_dir, "smoothed")
    
    # 결과 저장할 디렉토리 생성
    os.makedirs(dst_dir, exist_ok=True)
    
    class_dirs = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]
    print(f"총 {len(class_dirs)}개 클래스 처리 예정")
    
    total_files = sum(len([f for f in os.listdir(os.path.join(src_dir, d)) if f.endswith('.npz')]) for d in class_dirs)
    processed_files = 0
    
    for class_name in tqdm(class_dirs, desc="클래스 처리 중"):
        src_class_dir = os.path.join(src_dir, class_name)
        dst_class_dir = os.path.join(dst_dir, class_name)
        os.makedirs(dst_class_dir, exist_ok=True)
        
        npz_files = [f for f in os.listdir(src_class_dir) if f.endswith('.npz')]
        
        for npz_file in tqdm(npz_files, desc=f"{class_name} 처리 중", leave=False):
            src_path = os.path.join(src_class_dir, npz_file)
            dst_path = os.path.join(dst_class_dir, npz_file)
            
            if os.path.exists(dst_path):
                processed_files += 1
                continue
                
            try:
                data = np.load(src_path, allow_pickle=True)
                kp = data["keypoints"]
                
                # 메디안→가우시안 필터 적용
                kp_smoothed = smooth_keypoints(kp, median_kernel=5, gaussian_sigma=1.0)
                
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
            
            print(f"\r전체 진행률: {processed_files}/{total_files} ({processed_files/total_files*100:.1f}%)", end="")

if __name__ == "__main__":
    process_all_keypoints()
    print("\n모든 처리가 완료되었습니다.")
