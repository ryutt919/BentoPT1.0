import os
import subprocess

def launch_latest_tensorboard():
    # 1) 작업 디렉터리(학습 결과가 들어있는) 경로
    base_dir = r"C:\Users\kimt9\Desktop\RyuTTA\2025_3_1\ComputerVision\TermP\mmaction2\work_dirs\custom_stgcnpp"

    # 2) base_dir 아래 있는 모든 하위 폴더를 가져와서, 디렉터리인 것만 필터링
    #    (폴더 이름 예: "20250607_022504" 같은 타임스탬프 형식)
    all_subdirs = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]

    if not all_subdirs:
        print(f"[오류] '{base_dir}' 경로에 하위 폴더가 없습니다.")
        return

    # 3) 폴더 이름을 문자열 비교로 정렬하면, 타임스탬프 형식(YYYYMMDD_HHMMSS)이므로 가장 마지막 항목이 최신임
    all_subdirs.sort()
    latest_name = all_subdirs[-1]  # 가장 뒤에 오는(가장 큰) 이름

    # 4) 최신 디렉터리 전체 경로
    latest_dir = os.path.join(base_dir, latest_name)

    # 5) TensorBoard를 실행할 때 사용할 logdir을 지정
    #    보통 mmaction2나 PyTorch 학습 스크립트를 돌리면, 해당 위치에 이벤트 파일(events.out.tfevents.*)이 생성됨
    logdir = latest_dir

    # 6) TensorBoard 명령어 구성 (포트는 필요에 따라 변경 가능)
    tb_command = [
        "tensorboard",
        f"--logdir={logdir}",
        "--port=6006"
    ]

    # 7) 콘솔 출력으로 어느 경로를 잡았는지 확인
    print(f"[INFO] 최신 학습 로그 디렉터리: {latest_dir}")
    print(f"[INFO] 다음 명령어로 TensorBoard 실행:")
    print("       " + " ".join(tb_command))

    # 8) TensorBoard 서버를 백그라운드로 실행
    try:
        # Linux/macOS는 shell=False 권장, Windows도 동일
        tb_proc = subprocess.Popen(tb_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"[INFO] TensorBoard 프로세스 시작 (PID: {tb_proc.pid})")
    except Exception as e:
        print(f"[오류] TensorBoard 실행 중 예외 발생:\n{e}")

if __name__ == "__main__":
    launch_latest_tensorboard()
