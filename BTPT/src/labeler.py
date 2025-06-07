import os
import cv2
import numpy as np
import csv
from run_train import train_class_name

def visualize_keypoints_on_frame(frame, kpts, keypoint_names, skeleton_links):
    """
    한 프레임 위에 키포인트와 스켈레톤을 그리는 유틸 함수
    """
    # 컬러 정의 (BGR)
    ORANGE = (0, 165, 255)   # 왼쪽 관절
    GREEN = (0, 255, 0)      # 오른쪽 관절
    WHITE = (255, 255, 255)  # 테두리 및 연결선
    RED = (0, 0, 255)        # 발목-무릎 수직선

    # 오버레이 레이어 생성
    overlay = frame.copy()
    
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

        cv2.circle(overlay, (x_int, y_int), 5, circle_color, -1)
        cv2.circle(overlay, (x_int, y_int), 5, WHITE, 2)

    # 2) 스켈레톤 링크마다 선 그리기
    for (i, j) in skeleton_links:
        if kpts[i, 2] > 0 and kpts[j, 2] > 0:
            x1, y1 = int(kpts[i, 0]), int(kpts[i, 1])
            x2, y2 = int(kpts[j, 0]), int(kpts[j, 1])
            cv2.line(overlay, (x1, y1), (x2, y2), WHITE, 2)

    # 3) 발목에서 무릎 높이까지 수직선 그리기
    left_ankle_idx = 11   # 왼쪽 발목 인덱스
    right_ankle_idx = 12  # 오른쪽 발목 인덱스
    left_knee_idx = 9     # 왼쪽 무릎 인덱스
    right_knee_idx = 10   # 오른쪽 무릎 인덱스

    # 왼쪽 발목-무릎 수직선
    if kpts[left_ankle_idx, 2] > 0 and kpts[left_knee_idx, 2] > 0:
        ankle_x = int(kpts[left_ankle_idx, 0])
        ankle_y = int(kpts[left_ankle_idx, 1])
        knee_y = int(kpts[left_knee_idx, 1])
        cv2.line(overlay, (ankle_x, ankle_y), (ankle_x, knee_y), RED, 2)

    # 오른쪽 발목-무릎 수직선
    if kpts[right_ankle_idx, 2] > 0 and kpts[right_knee_idx, 2] > 0:
        ankle_x = int(kpts[right_ankle_idx, 0])
        ankle_y = int(kpts[right_ankle_idx, 1])
        knee_y = int(kpts[right_knee_idx, 1])
        cv2.line(overlay, (ankle_x, ankle_y), (ankle_x, knee_y), RED, 2)

    # 알파 블렌딩 적용
    alpha = 0.6  # 투명도 (0: 완전 투명, 1: 완전 불투명)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    

def label_and_visualize(video_path, npz_path, class_name, npz_file):
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
    save_dir = r"C:\Users\kimt9\Desktop\RyuTTA\2025_3_1\ComputerVision\TermP\BTPT\data"
    labels_dir = os.path.join(save_dir, 'labels', class_name)
    os.makedirs(labels_dir, exist_ok=True)
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    csv_path = os.path.join(labels_dir, f"{video_basename}.csv")

    LABEL_MAP_BY_CLASS = {
        "bench_pressing": {
            ord('w'): 'outlier',
            ord('a'): 'good',                 # 올바른 자세 (Good)
            ord('s'): 'elbow_not_vertical',    # 팔꿈치 수직 아님
            ord('d'): 'hand_position_bad',     # 손 위치 부적절
            ord('f'): 'elbow_flare',            # 팔꿈치가 몸통과 너무 벌어짐
            ord('g'): 'wrist_bent',            # 손목 꺾임
            ord('h'): 'shoulder_raise',        # 어깨 올라감
            ord('j'): 'grip_unstable',         # 그립 불안정
            ord('k'): 'spine_imbalance',       # 척추 불균형
            ord('l'): 'shoulder_asymmetry',    # 어깨 불균형
            ord(';'): 'barbell_unbalanced',    # 바벨 불균형 (임의로 ‘;’ 키 사용)
            ord('n'): 'max_contraction_needed',# 최대 수축 필요
            ord('m'): 'max_relaxation_needed', # 최대 이완 필요
        },
        "lunge": {
            ord('w'): 'outlier',
            ord('a'): 'good',                  # 올바른 자세 (Good)
            ord('d'): 'knee_over_toes',        # 무릎이 발끝을 넘어감
            ord('f'): 'back_round',            # 허리 굽음
            ord('j'): 'foot_unstable',         # 발 위치 불안정
            ord('l'): 'torso_shaking',         # 몸통 흔들림
            ord('x'): 'rear_knee_touch',       # 뒷다리 무릎 바닥에 닿음
            ord('c'): 'weight_shift_back',     # 무게중심 뒤로 쏠림
            ord('n'): 'max_contraction_needed',# 최대 수축 필요
            ord('m'): 'max_relaxation_needed', # 최대 이완 필요
        },
        "pull ups": {
            ord('w'): 'outlier',
            ord('a'): 'good',                  # 올바른 자세 (Good)
            ord('d'): 'shoulder_raise',        # 어깨 올라감
            ord('s'): 'grip_narrow',           # 그립 불안정
            ord('h'): 'torso_shaking',         # 몸통 흔들림
            ord('j'): 'spine_imbalance',       # 척추 불균형
            ord('k'): 'shoulder_asymmetry',    # 어깨 불균형
            ord('f'): 'max_contraction_needed',# 최대 수축 필요
            ord('g'): 'max_relaxation_needed', # 최대 이완 필요            
        },
        "push_up": {
            ord('w'): 'outlier',
            ord('a'): 'good',                  # 올바른 자세 (Good)
            ord('s'): 'elbow_not_vertical',    # 팔꿈치 수직 아님
            ord('d'): 'hand_position_bad',     # 손 위치 부적절
            ord('f'): 'elbow_flare',           # 팔꿈치가 몸통과 너무 벌어짐
            ord('g'): 'wrist_bent',            # 손목 꺾임
            ord('h'): 'shoulder_raise',        # 어깨 올라감
            ord('j'): 'grip_unstable',         # 그립 불안정
            ord('k'): 'neck_forward',          # 목 앞으로 내밀기
            ord('l'): 'torso_shaking',         # 몸통 흔들림
            ord('z'): 'spine_imbalance',       # 척추 불균형
            ord('x'): 'shoulder_asymmetry',    # 어깨 불균형
            ord('n'): 'max_contraction_needed',# 최대 수축 필요
            ord('m'): 'max_relaxation_needed', # 최대 이완 필요            
        },
        "squat": {
            ord('w'): 'outlier',
            ord('a'): 'good',                  # 올바른 자세 (Good)
            ord('s'): 'knee_over_toes',        # 무릎이 발끝을 넘어감
            ord('d'): 'back_round',            # 허리 굽음
            ord('f'): 'back_hyperextend',      # 허리 과신전
            ord('g'): 'knee_valgus',           # 무릎 내전
            ord('h'): 'foot_unstable',         # 발 위치 불안정
            ord('j'): 'torso_lean_forward',    # 상체 과도한 앞으로 숙임
            ord('k'): 'torso_shaking',         # 몸통 흔들림
            ord('l'): 'spine_imbalance',       # 척추 불균형
            ord('z'): 'weight_shift_back',     # 무게중심 뒤로 쏠림
            ord('x'): 'hip_excessive_move',    # 엉덩이 과도하게 올라감/내려감
            ord('c'): 'hip_asymmetry',         # 엉덩이 좌우 비대칭
            ord('n'): 'max_contraction_needed',# 최대 수축 필요
            ord('m'): 'max_relaxation_needed', # 최대 이완 필요              
        },
    }
    
    label_map = LABEL_MAP_BY_CLASS[class_name]

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

        cv2.putText(frame_resized, f" Record {idx+1}/{total_records}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)  


        # 7-4) 화면 출력 및 키 입력 대기
        cv2.imshow(f'{npz_file}', frame_resized)
        key = cv2.waitKey(0)  # 수동 진행: 키 입력 대기

        # 8) 키 처리 로직
        if key == 27:  # ESC: 전체 라벨링 종료
            print("사용자가 ESC를 눌러 라벨링을 종료했습니다.")
            break

        elif key == ord('i'): 
            idx = max(0, idx - 1)
            selected_labels.clear()
            print(f"이전 프레임으로 이동: Record {idx+1}/{total_records}")
            continue

        elif key == ord('p'):
            idx = min(total_records - 1, idx + 1)
            selected_labels.clear()
            print(f"다음 프레임으로 이동: Record {idx+1}/{total_records}")
            continue
        
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
        elif key == ord('w'):  # 'w': outlier
            print("outlier 라벨 선택")
            records.append((frame_num, class_name, ['outlier']))  # 리스트로 변경
            selected_labels.clear()  # 선택된 라벨 초기화
            idx += 1  # 다음 프레임으로
            continue

        elif key == ord('a'):  # 'a': good
            print("good 라벨 선택")
            records.append((frame_num, class_name, ['good']))  # 리스트로 변경
            selected_labels.clear()  # 선택된 라벨 초기화
            idx += 1  # 다음 프레임으로
            continue


        else:
            # 라벨 토글 키인지 확인
            if key in label_map:
                lbl = label_map[key]
                selected_labels.add(lbl)
                labels_list = sorted(selected_labels)
                records.append((frame_num, class_name, labels_list))
                print(f"라벨 선택: {lbl}")
                selected_labels.clear() 
            idx += 1  # 다음 프레임으로 이동
            
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

    import re

    def numeric_key(filename):
        # 숫자만 추출해서 int로 반환 (예: "10.npz" → 10)
        name = os.path.splitext(filename)[0]
        match = re.search(r'\d+', name)
        return int(match.group()) if match else float('inf')

    npz_files = sorted(
        [f for f in os.listdir(npz_dir) if f.endswith('.npz')],
        key=numeric_key
    )    
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

            label_and_visualize(video_path, npz_path, class_name, npz_file )
        except KeyboardInterrupt:
            print("\n사용자가 전체 라벨링을 중단했습니다.")
            break
        except Exception as e:
            print(f"에러 발생: {e}")
            continue

if __name__ == "__main__":
    # 예시: bench_pressing 클래스에 대해 라벨링 수행
    validate_keypoints_in_directory(train_class_name)
