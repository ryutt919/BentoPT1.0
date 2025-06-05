import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import os
from datetime import datetime
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

def process_video(video_path, class_name, file_count, frame_interval=5):
    """비디오에서 사람을 감지하고 포즈를 추정하여 키포인트 데이터만 저장"""
    # NPZ 파일 경로 확인
    output_dir = os.path.join(r'C:\Users\kimt9\Desktop\RyuTTA\2025_3_1\ComputerVision\TermP\mmaction2\data\kinetics400\keypoints', class_name)
    video_filename = os.path.basename(video_path)
    npz_filename = os.path.splitext(video_filename)[0] + '.npz'
    npz_output_path = os.path.join(output_dir, npz_filename)
    
    # 이미 처리된 파일이면 건너뛰기
    if os.path.exists(npz_output_path):
        print(f"\nSkipping {video_filename} - NPZ file already exists")
        return

    # YOLO 모델 로드
    detect_model = YOLO('yolov8x.pt')  # 객체 감지용
    pose_model = YOLO('yolov8x-pose.pt')  # 포즈 추정용
    
    # YOLO 출력 완전히 비활성화
    detect_model.verbose = False
    pose_model.verbose = False
    
    # 비디오 캡처 설정
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return
        
    # 비디오 정보 가져오기
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 키포인트 데이터 저장을 위한 리스트
    frame_numbers = []
    timestamps = []
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
    most_active_id = None  # 가장 활동적인 사람의 ID
    highest_confidence = 0  # 가장 높은 confidence 값
    
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
        detect_results = detect_model(frame, classes=[0], conf=0.5, verbose=False)[0]
        
        # 감지된 사람들의 포즈 추정
        pose_results = pose_model(frame, conf=0.5, verbose=False)[0]
        
        if len(detect_results.boxes) > 0 and pose_results.keypoints is not None:
            # numpy 배열로 변환
            kpts = pose_results.keypoints.data.cpu().numpy()
            
            # 각 감지된 사람에 대해
            for i, (box, keypoints) in enumerate(zip(detect_results.boxes, kpts)):
                # 바운딩 박스 좌표
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # 트래킹 ID 할당
                if i not in person_tracks:
                    person_tracks[i] = {
                        'box_history': [],
                        'keypoint_history': [],
                        'exercise_confidence': 0.0,
                        'last_frame': frame_count - frame_interval
                    }
                
                # 이전 프레임과의 간격이 너무 큰 경우 히스토리 초기화
                if frame_count - person_tracks[i]['last_frame'] > frame_interval * 2:
                    person_tracks[i]['box_history'] = []
                    person_tracks[i]['keypoint_history'] = []
                
                # 박스와 키포인트 히스토리 업데이트
                person_tracks[i]['box_history'].append([x1, y1, x2, y2])
                person_tracks[i]['keypoint_history'].append(keypoints)
                person_tracks[i]['last_frame'] = frame_count
                
                # 히스토리 크기 제한
                max_history = 6
                if len(person_tracks[i]['box_history']) > max_history:
                    person_tracks[i]['box_history'].pop(0)
                    person_tracks[i]['keypoint_history'].pop(0)
                
                # 운동 여부 판단
                if len(person_tracks[i]['keypoint_history']) > 2:
                    total_movement = 0.0
                    valid_keypoints = 0
                    
                    # 모든 주요 관절의 움직임 분석
                    for kp_name, kp_idx in EXERCISE_KEYPOINTS.items():
                        # 키포인트 유효성 검사 추가
                        if (kp_idx < len(keypoints) and 
                            kp_idx < len(person_tracks[i]['keypoint_history'][0]) and
                            len(keypoints[kp_idx]) >= 3 and 
                            len(person_tracks[i]['keypoint_history'][0][kp_idx]) >= 3 and
                            keypoints[kp_idx][2] > 0.5 and  # 현재 프레임의 신뢰도
                            person_tracks[i]['keypoint_history'][0][kp_idx][2] > 0.5):  # 이전 프레임의 신뢰도
                            # 관절의 이동 거리 계산
                            movement = np.linalg.norm(
                                person_tracks[i]['keypoint_history'][-1][kp_idx][:2] -
                                person_tracks[i]['keypoint_history'][0][kp_idx][:2]
                            )
                            total_movement += movement
                            valid_keypoints += 1
                    
                    # 유효한 관절이 있는 경우에만 평균 움직임 계산
                    if valid_keypoints > 0:
                        avg_movement = total_movement / valid_keypoints
                        
                        # 운동 여부 판단을 위한 임계값
                        threshold = 15.0
                        confidence_update = 0.3 if avg_movement > threshold else -0.2
                        
                        # 운동 신뢰도 업데이트 (상한 없음)
                        person_tracks[i]['exercise_confidence'] = max(0.0,
                            person_tracks[i]['exercise_confidence'] + confidence_update)
                        
                        # 가장 높은 confidence를 가진 사람 업데이트
                        if person_tracks[i]['exercise_confidence'] > highest_confidence:
                            highest_confidence = person_tracks[i]['exercise_confidence']
                            most_active_id = i
            
            # 가장 활동적인 사람의 키포인트만 저장
            if most_active_id is not None and most_active_id < len(kpts):
                current_keypoints = np.zeros((len(SELECTED_KEYPOINTS), 3))
                keypoints = kpts[most_active_id]
                
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
                    keypoints_data.append(current_keypoints)
    
    cap.release()
    
    # 키포인트 데이터를 NPZ 파일로 저장
    if keypoints_data:
        # 저장 경로 설정
        output_dir = os.path.join(r'C:\Users\kimt9\Desktop\RyuTTA\2025_3_1\ComputerVision\TermP\mmaction2\data\kinetics400\keypoints', class_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # 파일명에서 .mp4 확장자를 제거하고 .npz로 변경
        video_filename = os.path.basename(video_path)
        npz_filename = os.path.splitext(video_filename)[0] + '.npz'
        npz_output_path = os.path.join(output_dir, npz_filename)
        
        np.savez_compressed(
            npz_output_path,
            frame_numbers=np.array(frame_numbers),
            timestamps=np.array(timestamps),
            keypoints=np.array(keypoints_data),
            keypoint_names=KEYPOINT_NAMES,
            video_info=np.array([
                os.path.basename(video_path),
                str(fps),
                str(width),
                str(height),
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                str(frame_interval)
            ], dtype=object)
        )
        print(f"\nKeypoints data saved to: {npz_output_path}")
        print(f"Processed frames: {len(frame_numbers)}, Frame interval: {frame_interval}")
        print(f"Final confidence of most active person: {highest_confidence:.2f}")
    else:
        print("\nNo exercise movements detected in the video.")

def main():
    video_dir = r'C:\Users\kimt9\Desktop\RyuTTA\2025_3_1\ComputerVision\TermP\mmaction2\data\kinetics400\videos'
    class_names = [d for d in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, d))]
    print(f"총 {len(class_names)}개 클래스 처리 예정")
    
    # 전체 처리해야 할 비디오 파일 수 계산 (이미 처리된 파일 제외)
    total_videos = 0
    remaining_videos = 0
    for class_name in class_names:
        class_dir = os.path.join(video_dir, class_name)
        video_files = [f for f in os.listdir(class_dir) if f.endswith('.mp4')]
        total_videos += len(video_files)
        
        # 이미 처리된 파일 확인
        output_dir = os.path.join(r'C:\Users\kimt9\Desktop\RyuTTA\2025_3_1\ComputerVision\TermP\mmaction2\data\kinetics400\keypoints', class_name)
        for video_file in video_files:
            npz_file = os.path.splitext(video_file)[0] + '.npz'
            npz_path = os.path.join(output_dir, npz_file)
            if not os.path.exists(npz_path):
                remaining_videos += 1
    
    print(f"총 비디오 파일 수: {total_videos}")
    print(f"이미 처리된 파일 수: {total_videos - remaining_videos}")
    print(f"처리해야 할 파일 수: {remaining_videos}")
    
    # 2개의 프로세스 사용
    num_processes = 2
    print(f"사용할 CPU 코어 수: {num_processes}")
    
    # 멀티프로세싱 풀 생성
    pool = mp.Pool(processes=num_processes)
    
    try:
        # 각 클래스별로 비디오 처리 함수를 병렬로 실행
        process_func = partial(process_video, video_dir=video_dir)
        list(tqdm(pool.imap(process_func, class_names), total=len(class_names), desc="클래스 처리 중"))
    finally:
        pool.close()
        pool.join()

if __name__ == "__main__":
    # Windows에서 멀티프로세싱 사용 시 필요한 보호 코드
    mp.freeze_support()
    main() 