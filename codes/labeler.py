import os
import cv2
import numpy as np
import csv

def visualize_keypoints_on_frame(frame, kpts, keypoint_names, skeleton_links):
    """
    한 프레임 위에 키포인트와 스켈레톤을 그리는 유틸 함수
    Args:
        frame: 원본 프레임 이미지 (numpy 배열)
        kpts: 해당 프레임의 키포인트 배열 (13, 3)
        keypoint_names: 키포인트 이름 리스트
        skeleton_links: 스켈레톤 연결 쌍 리스트
    """
    # 컬러 정의 (BGR)
    ORANGE = (0, 165, 255)   # 왼쪽 관절
    GREEN  = (0, 255, 0)     # 오른쪽 관절
    WHITE  = (255, 255, 255) # 테두리 및 연결선

    # 1) 키포인트마다 원 그리기
    for kp_idx, (x, y, conf) in enumerate(kpts):
        if conf <= 0:
            continue
        x_int, y_int = int(x), int(y)
        name = keypoint_names[kp_idx].decode('utf-8') if isinstance(keypoint_names[kp_idx], bytes) else keypoint_names[kp_idx]
        if name.startswith('left_'):
            circle_color = ORANGE
        elif name.startswith('right_'):
            circle_color = GREEN
        else:
            circle_color = WHITE

        cv2.circle(frame, (x_int, y_int), 5, circle_color, -1)
        cv2.circle(frame, (x_int, y_int), 5, WHITE, 2)

    # 2) 스켈레톤 링크마다 선 그리기
    for (i, j) in skeleton_links:
        if kpts[i, 2] > 0 and kpts[j, 2] > 0:
            x1, y1 = int(kpts[i, 0]), int(kpts[i, 1])
            x2, y2 = int(kpts[j, 0]), int(kpts[j, 1])
            cv2.line(frame, (x1, y1), (x2, y2), WHITE, 2)

def label_and_visualize(video_path, npz_path, class_name):
    """
    NPZ 키포인트를 시각화하면서 한 프레임씩 라벨링을 수행합니다.
    - 스페이스바: 현재 선택된 라벨들을 저장(라벨이 있을 때만)하고 다음 프레임으로 이동
    - Backspace: 가장 최근 저장한 프레임 레코드를 삭제하고 해당 프레임으로 돌아감
    - 라벨 토글: 지정된 키(0~9, a,k,p,q,r,t,u,v,w,x,y,z)를 눌러 토글
    - ESC: 라벨링 중지
    Args:
        video_path: 원본 비디오 파일 경로
        npz_path: NPZ 파일 경로
        class_name: 현재 클래스 이름 (CSV에 함께 저장)
    """
    # 1) 경로 확인
    if not os.path.exists(video_path):
        print(f"Error: 영상 파일을 찾을 수 없습니다 → {video_path}")
        return
    if not os.path.exists(npz_path):
        print(f"Error: NPZ 파일을 찾을 수 없습니다 → {npz_path}")
        return

    # 2) NPZ 로드
    data = np.load(npz_path, allow_pickle=True)
    frame_numbers = data['frame_numbers']      # (N,)
    keypoints_all = data['keypoints']          # (N, 13, 3)
    keypoint_names = data['keypoint_names']    # ['nose', 'left_shoulder', ...]
    video_info = data['video_info']            # [filename, fps, width, height, ...]

    # 3) 스켈레톤 연결 정의
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

    # 4) 라벨 키맵 정의 (키코드 → 라벨 문자열)
    label_map = {
        ord('0'): 'outlier',
        ord('1'): 'good',
        ord('2'): 'elbow_not_vertical',
        ord('3'): 'knee_over_toes',
        ord('4'): 'back_round',
        ord('5'): 'back_hyperextend',
        ord('6'): 'knee_valgus',
        ord('7'): 'foot_unstable',
        ord('8'): 'torso_lean_forward',
        ord('9'): 'torso_shaking',
        ord('a'): 'hand_position_bad',
        ord('k'): 'elbow_flare',
        ord('p'): 'rear_knee_touch',
        ord('q'): 'wrist_bent',
        ord('r'): 'shoulder_raise',
        ord('t'): 'grip_unstable',
        ord('u'): 'hip_excessive_move',
        ord('v'): 'spine_imbalance',
        ord('w'): 'neck_forward',
        ord('x'): 'hip_asymmetry',
        ord('y'): 'weight_shift_back',
        ord('z'): 'shoulder_asymmetry'
    }

    # 5) 비디오 열기
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: 영상 파일을 열 수 없습니다 → {video_path}")
        return

    total_records = keypoints_all.shape[0]
    idx = 0  # 현재 인덱스
    records = []  # 저장된 레코드: 리스트 of (frame_num, class_name, [라벨 리스트])
    selected_labels = set()  # 현재 프레임에서 선택된 라벨

    # 6) 라벨 저장할 CSV 디렉토리/파일 경로 설정
    base_dir = r"C:\Users\kimt9\Desktop\RyuTTA\2025_3_1\ComputerVision\TermP\mmaction2\data\kinetics400"
    labels_dir = os.path.join(base_dir, 'labels', class_name)
    os.makedirs(labels_dir, exist_ok=True)
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    csv_path = os.path.join(labels_dir, f"{video_basename}.csv")

    # 7) 메인 루프
    while idx < total_records:
        frame_num = int(frame_numbers[idx])
        kpts = keypoints_all[idx]

        # 7-1) 해당 프레임으로 이동 후 읽기
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Frame {frame_num}을 읽을 수 없습니다. 건너뜁니다.")
            idx += 1
            continue

        # 7-2) 키포인트 오버레이
        visualize_keypoints_on_frame(frame, kpts, keypoint_names, skeleton_links)

        # 7-3) 화면 하단에 선택된 라벨 텍스트 표시
        labels_text = "slected: [" + ", ".join(sorted(selected_labels)) + "]"
        cv2.putText(frame, labels_text, (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        height, width = frame.shape[:2]
        max_dim = max(width, height)
        scale = 800 / max_dim
        new_w = int(width * scale)
        new_h = int(height * scale)
        frame_resized = cv2.resize(frame, (new_w, new_h))


        # 7-4) 화면 출력 및 키 입력 대기
        cv2.imshow('labeling', frame_resized)
        key = cv2.waitKey(0)  # 수동 진행: 키 입력 대기

        # 8) 키 처리 로직
        if key == 27:  # ESC: 전체 라벨링 종료
            print("사용자가 ESC를 눌러 라벨링을 종료했습니다.")
            break

        elif key == 8:  # Backspace: 가장 최근 레코드 삭제 후 이전 프레임으로 이동
            if records:
                last_frame, _, _ = records.pop()
                idx = max(0, idx - 1)  # 이전 인덱스로 이동
                selected_labels.clear()  # 토글 상태 초기화
                print(f"최근 저장된 프레임 {last_frame} 레코드를 삭제했습니다.")
            else:
                print("삭제할 레코드가 없습니다.")
            continue

        elif key == 32:  # Spacebar: 현재 프레임 레코드 저장 후 다음으로 이동
            # *** 수정된 부분: selected_labels가 비어 있으면 기록하지 않음 ***
            if selected_labels:
                labels_list = sorted(selected_labels)
                records.append((frame_num, class_name, labels_list))
                print(f"프레임 {frame_num} 저장: {labels_list}")
            else:
                print(f"프레임 {frame_num}에는 라벨이 없어 저장하지 않습니다.")
            selected_labels.clear()
            idx += 1
            continue

        else:
            # 라벨 토글 키인지 확인
            if key in label_map:
                lbl = label_map[key]
                if lbl in selected_labels:
                    selected_labels.remove(lbl)
                    print(f"라벨 해제: {lbl}")
                else:
                    selected_labels.add(lbl)
                    print(f"라벨 선택: {lbl}")
            # 그 외 키는 무시
            continue

    # 9) 루프 종료 후 CSV로 저장 (records에 저장된 항목만)
    if records:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['frame', 'class', 'labels'])
            for frame_num, cls, lbls in records:
                lbls_str = ";".join(lbls)
                writer.writerow([frame_num, cls, lbls_str])
        print(f"라벨링 결과를 CSV로 저장했습니다 → {csv_path}")
    else:
        print("저장된 레코드가 없어 CSV를 생성하지 않습니다.")

    cap.release()
    cv2.destroyAllWindows()

def validate_keypoints_in_directory(class_name):
    """
    특정 클래스의 모든 NPZ 파일에 대해 라벨링 모드를 실행합니다.
    Args:
        class_name: 'bench_pressing', 'squat' 등
    """
    base_dir = r"C:\Users\kimt9\Desktop\RyuTTA\2025_3_1\ComputerVision\TermP\mmaction2\data\kinetics400"
    video_dir = os.path.join(base_dir, "videos", class_name)
    npz_dir = os.path.join(base_dir, "smoothed", class_name)

    if not os.path.isdir(video_dir) or not os.path.isdir(npz_dir):
        print(f"Error: 디렉토리를 확인하세요 → {video_dir}, {npz_dir}")
        return

    npz_files = [f for f in os.listdir(npz_dir) if f.endswith('.npz')]
    total_files = len(npz_files)

    print(f"\n{class_name} 클래스의 총 {total_files}개 NPZ 파일을 라벨링합니다.")
    print("조작법:")
    print("- 숫자(0~9), 문자(a,k,p,q,r,t,u,v,w,x,y,z) 키: 해당 라벨 토글")
    print("- Spacebar: 선택된 라벨 저장(라벨이 있을 때만) 후 다음 프레임으로 이동")
    print("- Backspace: 최근 저장한 프레임 레코드 삭제 후 해당 프레임으로 복귀")
    print("- ESC: 현재 파일 라벨링 중단 및 다음 파일로 넘어감\n")

    for idx, npz_file in enumerate(npz_files, 1):
        video_file = os.path.splitext(npz_file)[0] + '.mp4'
        video_path = os.path.join(video_dir, video_file)
        npz_path = os.path.join(npz_dir, npz_file)

        print(f"[{idx}/{total_files}] 라벨링: {npz_file}")
        try:
            label_and_visualize(video_path, npz_path, class_name)
        except KeyboardInterrupt:
            print("\n사용자가 전체 라벨링을 중단했습니다.")
            break
        except Exception as e:
            print(f"에러 발생: {e}")
            continue

if __name__ == "__main__":
    # 예시: bench_pressing 클래스에 대해 라벨링 수행
    validate_keypoints_in_directory("bench_pressing")
