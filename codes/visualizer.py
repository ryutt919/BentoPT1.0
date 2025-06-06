import os
import cv2
import numpy as np

def visualize_keypoints_on_video(video_path, npz_path, auto_play=False):
    """
    저장된 NPZ 파일에 들어 있는 키포인트들을 원본 영상 위에 오버레이하여
    왼쪽 관절은 주황색, 오른쪽 관절은 초록색, 원 테두리는 흰색, 관절 연결은 흰색 선으로 표시합니다.
    한 프레임씩 화면에 띄우며, 아무 키나 누르면 다음 프레임으로 넘어갑니다.
    
    Args:
        video_path: 원본 비디오 경로
        npz_path: NPZ 파일 경로
        auto_play: True이면 자동 재생 (스페이스바로 일시정지 가능), False이면 수동 진행
    """

    # 1) 경로 존재 여부 확인
    if not os.path.exists(video_path):
        print(f"Error: 영상 파일을 찾을 수 없습니다 → {video_path}")
        return
    if not os.path.exists(npz_path):
        print(f"Error: NPZ 파일을 찾을 수 없습니다 → {npz_path}")
        return

    # 2) NPZ 파일 로드
    data = np.load(npz_path, allow_pickle=True)
    frame_numbers = data['frame_numbers']      # shape: (N,)
    keypoints_all = data['keypoints']         # shape: (N, 13, 3)
    keypoint_names = data['keypoint_names']   # ['nose', 'left_shoulder', ..., 'right_ankle']
    video_info = data['video_info']           # [filename, fps, width, height, timestamp, frame_interval]

    # 3) 스켈레톤 연결 쌍 정의 (인덱스 기준)
    # keypoint_names 순서: 
    # 0: nose
    # 1: left_shoulder
    # 2: right_shoulder
    # 3: left_elbow
    # 4: right_elbow
    # 5: left_wrist
    # 6: right_wrist
    # 7: left_hip
    # 8: right_hip
    # 9: left_knee
    # 10: right_knee
    # 11: left_ankle
    # 12: right_ankle

    skeleton_links = [
        (1, 3),   # left_shoulder ↔ left_elbow
        (3, 5),   # left_elbow ↔ left_wrist
        (2, 4),   # right_shoulder ↔ right_elbow
        (4, 6),   # right_elbow ↔ right_wrist
        (1, 7),   # left_shoulder ↔ left_hip
        (2, 8),   # right_shoulder ↔ right_hip
        (7, 9),   # left_hip ↔ left_knee
        (9, 11),  # left_knee ↔ left_ankle
        (8, 10),  # right_hip ↔ right_knee
        (10, 12), # right_knee ↔ right_ankle
        (7, 8)    # left_hip ↔ right_hip
    ]

    # 4) 컬러 정의 (BGR 형식)
    ORANGE = (0, 165, 255)   # 왼쪽 관절
    GREEN  = (0, 255, 0)     # 오른쪽 관절
    WHITE  = (255, 255, 255) # 테두리 및 연결선

    # 5) 영상 열기
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: 영상 파일을 열 수 없습니다 → {video_path}")
        return

    # 6) 프레임 단위로 오버레이
    total_records = keypoints_all.shape[0]
    for idx in range(total_records):
        frame_num = int(frame_numbers[idx])
        kpts = keypoints_all[idx]   # shape: (13, 3)

        # 6-1) 해당 프레임으로 이동
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Frame {frame_num}을 읽을 수 없습니다. 건너뜁니다.")
            continue

        # 6-2) 모든 키포인트에 대해 원 그리기
        for kp_idx, (x, y, conf) in enumerate(kpts):
            if conf <= 0:  # 信頼도가 0 이하면 건너뜀
                continue
            x_int, y_int = int(x), int(y)

            # 왼쪽/오른쪽 판정
            name = keypoint_names[kp_idx].decode('utf-8') if isinstance(keypoint_names[kp_idx], bytes) else keypoint_names[kp_idx]
            if name.startswith('left_'):
                circle_color = ORANGE
            elif name.startswith('right_'):
                circle_color = GREEN
            else:
                circle_color = WHITE  # nose 등의 중앙 관절은 흰색으로 표시

            # 원(채워진) → 테두리(두께 2) 순서로 그리면 테두리가 깔끔하게 분리됨
            cv2.circle(frame, (x_int, y_int), 5, circle_color, -1)      # 반지름=5, filled
            cv2.circle(frame, (x_int, y_int), 5, WHITE, 2)              # 반지름=5, 테두리 두께=2

        # 6-3) 스켈레톤 링크에 따라 흰색 선으로 연결
        for (i, j) in skeleton_links:
            # 두 관절 모두 confidence > 0 일 때만 선 표시
            if kpts[i, 2] > 0 and kpts[j, 2] > 0:
                x1, y1 = int(kpts[i, 0]), int(kpts[i, 1])
                x2, y2 = int(kpts[j, 0]), int(kpts[j, 1])
                cv2.line(frame, (x1, y1), (x2, y2), WHITE, 2)

        # 7) 프레임 번호 및 키포인트 수 텍스트 (선택 사항)
        cv2.putText(frame, f"Frame: {frame_num} / Record {idx+1}/{total_records}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)

        # 8) 화면 출력 → auto_play 모드에 따라 다르게 동작
        cv2.imshow('Keypoints 확인', frame)
        if auto_play:
            key = cv2.waitKey(0            )  # 30ms 대기 (약 33fps)
        else:
            key = cv2.waitKey(0)  # 키 입력 대기

        # ESC: 종료, 스페이스바: 재생/일시정지
        if key == 27:  # ESC
            break
        elif key == 32:  # 스페이스바
            auto_play = not auto_play

    cap.release()
    cv2.destroyAllWindows()
    return key  # ESC 키가 눌렸는지 확인하기 위해 반환

def validate_keypoints_in_directory(class_name):
    """특정 클래스의 모든 NPZ 파일들을 순차적으로 검증합니다."""
    # 기본 경로 설정
    base_dir = r"C:\Users\kimt9\Desktop\RyuTTA\2025_3_1\ComputerVision\TermP\mmaction2\data\kinetics400"
    video_dir = os.path.join(base_dir, "videos", class_name)
    npz_dir = os.path.join(base_dir, "smoothed", class_name)

    # NPZ 파일 목록 가져오기
    npz_files = [f for f in os.listdir(npz_dir) if f.endswith('.npz')]
    total_files = len(npz_files)
    
    print(f"\n{class_name} 클래스의 총 {total_files}개 NPZ 파일을 검증합니다.")
    print("조작법:")
    print("- 스페이스바: 자동 재생 / 일시정지")
    print("- ESC: 현재 파일 건너뛰기")
    print("- Ctrl+C: 프로그램 종료")
    
    for idx, npz_file in enumerate(npz_files, 1):
        # 비디오 파일명 (확장자만 다름)
        video_file = os.path.splitext(npz_file)[0] + '.mp4'
        
        # 전체 경로
        video_path = os.path.join(video_dir, video_file)
        npz_path = os.path.join(npz_dir, npz_file)
        
        print(f"\n[{idx}/{total_files}] 검증 중: {npz_file}")
        
        try:
            # 키포인트 시각화 (자동 재생 모드로 시작)
            key = visualize_keypoints_on_video(video_path, npz_path, auto_play=False)
            
            # ESC 키가 눌렸다면 다음 파일로
            if key == 27:
                print(f"사용자가 {npz_file} 검증을 건너뛰었습니다.")
                continue
                
        except KeyboardInterrupt:
            print("\n사용자가 검증을 중단했습니다.")
            break
        except Exception as e:
            print(f"에러 발생: {e}")
            continue

if __name__ == "__main__":
    # 특정 클래스의 모든 NPZ 파일 검증
    validate_keypoints_in_directory("bench_pressing")
                 