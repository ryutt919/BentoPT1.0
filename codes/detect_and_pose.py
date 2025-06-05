import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import os
from datetime import datetime
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import logging

# -----------------------------------
# 1. 전역 모델 로딩 (한 번만)
# -----------------------------------
DETECT_MODEL = YOLO('yolov8x.pt')
POSE_MODEL   = YOLO('yolov8x-pose.pt')
DETECT_MODEL.verbose = False
POSE_MODEL.verbose   = False

# -----------------------------------
# 2. 로깅 설정
# -----------------------------------
logging.basicConfig(
    filename='app.log',
    filemode='a',  
    format='%(asctime)s %(levelname)s: %(message)s',
    level=logging.INFO
)


def process_video(video_path, class_name, file_count, frame_interval=3):
    """비디오에서 사람을 감지하고 포즈를 추정하여 키포인트 데이터만 저장"""
    # NPZ 파일 경로 확인
    output_dir = os.path.join(
        r'C:\Users\kimt9\Desktop\RyuTTA\2025_3_1\ComputerVision\TermP\mmaction2\data\kinetics400\keypoints',
        class_name
    )
    video_filename = os.path.basename(video_path)
    npz_filename = os.path.splitext(video_filename)[0] + '.npz'
    npz_output_path = os.path.join(output_dir, npz_filename)
    
    # 이미 처리된 파일이면 건너뛰기
    if os.path.exists(npz_output_path):
        logging.info(f"Skipping {video_filename} - NPZ file already exists")
        return

    # 비디오 캡처 설정
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Cannot open video file {video_path}")
        return
        
    # 비디오 정보 가져오기
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 키포인트 데이터 저장을 위한 리스트
    frame_numbers = []
    timestamps = []
    person_ids = []
    keypoints_data = []
    
    # YOLO Pose의 키포인트 매핑 정의
    SELECTED_KEYPOINTS = {
        'nose': 0,
        'left_shoulder': 5,
        'right_shoulder': 6,
        'left_elbow': 7,
        'right_elbow': 8,
        'left_wrist': 9,
        'right_wrist': 10,
        'left_hip': 11,
        'right_hip': 12,
        'left_knee': 13,
        'right_knee': 14,
        'left_ankle': 15,
        'right_ankle': 16
    }
    
    # 운동 감지를 위한 주요 관절 인덱스
    EXERCISE_KEYPOINTS = {
        'left_shoulder': 5,
        'right_shoulder': 6,
        'left_elbow': 7,
        'right_elbow': 8,
        'left_hip': 11,
        'right_hip': 12,
        'left_knee': 13,
        'right_knee': 14
    }
    
    # NPZ 파일에 저장될 키포인트 이름 순서
    KEYPOINT_NAMES = list(SELECTED_KEYPOINTS.keys())
    
    frame_count = 0
    person_tracks = {}  # 사람별 트래킹 정보
    
    # 프레임 범위 생성
    frame_range = range(0, total_frames, frame_interval)
    
    # tqdm으로 진행 상황 표시
    for frame_count in tqdm(frame_range, desc='Processing frames', dynamic_ncols=True, leave=True):
        # 프레임 위치 설정
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read()
        if not ret:
            break
        
        # 사람 감지
        detect_results = DETECT_MODEL(frame, classes=[0], conf=0.5, verbose=False)[0]
        
        # 두 명 이상이 감지되면 이 영상은 건너뛰기
        if len(detect_results.boxes) > 1:
            logging.info(f"Skipping {video_filename} - Multiple people detected ({len(detect_results.boxes)} people)")
            cap.release()
            return
            
        # 감지된 사람이 없으면 다음 프레임으로
        if len(detect_results.boxes) == 0:
            continue
        
        # 감지된 사람의 포즈 추정
        pose_results = POSE_MODEL(frame, conf=0.5, verbose=False)[0]
        
        if pose_results.keypoints is not None:
            # numpy 배열로 변환
            kpts = pose_results.keypoints.data.cpu().numpy()
            keypoints = kpts[0]  # 한 명만 있으므로 첫 번째 사람의 키포인트
            
            # 현재 프레임의 키포인트 저장
            current_keypoints = np.zeros((len(SELECTED_KEYPOINTS), 3))
            for idx, (kp_name, yolo_idx) in enumerate(SELECTED_KEYPOINTS.items()):
                if yolo_idx < len(keypoints):
                    kp = keypoints[yolo_idx]
                    if not np.any(np.isnan(kp)):
                        x, y, conf = kp
                        if conf > 0.5:
                            current_keypoints[idx] = [x, y, conf]
            
            if np.any(current_keypoints[:, 2] > 0.5):
                frame_numbers.append(frame_count)
                timestamps.append(frame_count / fps)
                person_ids.append(0)  # 항상 ID는 0으로 설정 (한 명만 처리하므로)
                keypoints_data.append(current_keypoints)
    
    cap.release()
    
    # 키포인트 데이터를 NPZ 파일로 저장
    if keypoints_data:
        # 저장 경로 설정
        os.makedirs(output_dir, exist_ok=True)
        
        np.savez_compressed(
            npz_output_path,
            frame_numbers=np.array(frame_numbers),
            timestamps=np.array(timestamps),
            person_ids=np.array(person_ids),
            keypoints=np.array(keypoints_data),
            keypoint_names=np.array(KEYPOINT_NAMES),
            video_info=np.array([
                os.path.basename(video_path),
                str(fps),
                str(width),
                str(height),
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                str(frame_interval)
            ], dtype=object)
        )
        logging.info(f"Keypoints data saved to: {npz_output_path} ({len(frame_numbers)} frames)")
    else:
        logging.warning(f"No valid keypoints detected in {video_filename}")

def process_class_videos(class_name, video_dir):
    """한 클래스의 비디오들을 처리"""
    class_dir = os.path.join(video_dir, class_name)
    if not os.path.isdir(class_dir):
        return
    
    video_files = [f for f in os.listdir(class_dir) if f.endswith('.mp4')]
    for file_count, filename in enumerate(video_files, 1):
        video_path = os.path.join(class_dir, filename)
        process_video(video_path, class_name, file_count, frame_interval=3)

def main():
    video_dir = r'C:\Users\kimt9\Desktop\RyuTTA\2025_3_1\ComputerVision\TermP\mmaction2\data\kinetics400\videos'
    class_names = [d for d in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, d))]
    logging.info(f"총 {len(class_names)}개 클래스 처리 예정")
    
    # 전체 처리해야 할 비디오 파일 수 계산 (이미 처리된 파일 제외)
    total_videos = 0
    remaining_videos = 0
    for class_name in class_names:
        class_dir = os.path.join(video_dir, class_name)
        video_files = [f for f in os.listdir(class_dir) if f.endswith('.mp4')]
        total_videos += len(video_files)
        
        # 이미 처리된 파일 확인
        output_dir = os.path.join(
            r'C:\Users\kimt9\Desktop\RyuTTA\2025_3_1\ComputerVision\TermP\mmaction2\data\kinetics400\keypoints',
            class_name
        )
        for video_file in video_files:
            npz_file = os.path.splitext(video_file)[0] + '.npz'
            npz_path = os.path.join(output_dir, npz_file)
            if not os.path.exists(npz_path):
                remaining_videos += 1
    
    logging.info(f"총 비디오 파일 수: {total_videos}")
    logging.info(f"이미 처리된 파일 수: {total_videos - remaining_videos}")
    logging.info(f"처리해야 할 파일 수: {remaining_videos}")

    print(f"총 비디오 파일 수: {total_videos}")
    print(f"이미 처리된 파일 수: {total_videos - remaining_videos}")
    print(f"처리해야 할 파일 수: {remaining_videos}")
    
    # 2개의 프로세스 사용
    num_processes = 2
    logging.info(f"사용할 CPU 코어 수: {num_processes}")
    
    # 멀티프로세싱 풀 생성
    pool = mp.Pool(processes=num_processes)
    
    try:
        # 각 클래스별로 비디오 처리 함수를 병렬로 실행
        process_func = partial(process_class_videos, video_dir=video_dir)
        list(tqdm(pool.imap(process_func, class_names), total=len(class_names), desc="클래스 처리 중"))
    finally:
        pool.close()
        pool.join()

if __name__ == "__main__":
    # Windows에서 멀티프로세싱 사용 시 필요한 보호 코드
    mp.freeze_support()
    main()
