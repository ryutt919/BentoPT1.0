import os
import sys

# MMAction2 경로 추가
mmaction_path = os.path.abspath("../mmaction2")
sys.path.append(mmaction_path)
train_class_name = 'pull_ups'

# 환경 변수 설정
if __name__ == "__main__":
    # 현재 작업 디렉토리를 BTPT 폴더로 변경
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # 명령줄 인자 설정
    sys.argv = [
        sys.argv[0],  # 스크립트 이름
        'configs/custom_config.py',  # config 파일 경로
        '--work-dir', 'models/'+ train_class_name,  # 작업 디렉토리
        # '--seed', '42',  # 랜덤 시드
    ]
    
    # 선택적 인자 추가
    # sys.argv.extend(['--amp'])  # 자동 혼합 정밀도 학습
    # sys.argv.extend(['--auto-scale-lr'])  # 자동 학습률 조정
    # sys.argv.extend(['--resume'])  # 학습 재개
    
    # train.py의 main() 함수 호출
    from train import main
    main()