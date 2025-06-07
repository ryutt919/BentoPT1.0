import os
import numpy as np
from tqdm import tqdm


def normalize_keypoints(kp_array, width, height):
    """프레임 크기에 대해 [0, 1] 범위로 정규화."""
    width = width if width > 0 else 1.0
    height = height if height > 0 else 1.0
    normed = kp_array.copy().astype(float)
    normed[..., 0] /= width
    normed[..., 1] /= height
    return normed


def process_all_normalization():
    base_dir = r"C:\Users\kimt9\Desktop\RyuTTA\2025_3_1\ComputerVision\TermP\mmaction2\data\kinetics400"
    src_dir = os.path.join(base_dir, "smoothed")
    dst_dir = os.path.join(base_dir, "normalized")

    os.makedirs(dst_dir, exist_ok=True)

    class_dirs = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]
    print(f"총 {len(class_dirs)}개 클래스 처리 예정")

    total_files = sum(
        len([f for f in os.listdir(os.path.join(src_dir, d)) if f.endswith('.npz')])
        for d in class_dirs
    )
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

                width = float(data["video_info"][2])
                height = float(data["video_info"][3])

                kp_normed = normalize_keypoints(kp, width, height)

                np.savez_compressed(
                    dst_path,
                    frame_numbers=data["frame_numbers"],
                    timestamps=data["timestamps"],
                    person_ids=data["person_ids"],
                    keypoints=kp_normed,
                    keypoint_names=data["keypoint_names"],
                    video_info=data["video_info"],
                )
                processed_files += 1
            except Exception as e:
                print(f"\nError processing {src_path}: {str(e)}")
                continue

            print(
                f"\r전체 진행률: {processed_files}/{total_files} ({processed_files / total_files * 100:.1f}%)",
                end="",
            )


if __name__ == "__main__":
    process_all_normalization()
    print("\n정규화 완료")