# BTPT (Bench Press Training Pose Tracker)

벤치프레스 자세 분석 및 피드백 시스템

## 프로젝트 구조
```
BTPT/
├── configs/         # 모델 설정 파일
├── data/           # 데이터셋 및 라벨
├── models/         # 학습된 모델 체크포인트
├── notebooks/      # Jupyter 노트북
├── results/        # 실험 결과
├── src/           # 소스 코드
│   ├── train.py          # 모델 학습
│   ├── inference.py      # 모델 추론
│   └── custom_hooks.py   # 커스텀 훅
└── utils/         # 유틸리티 함수

## 설치 방법

```bash
# 가상환경 생성
python -m venv pose-env

# 가상환경 활성화
# Windows:
./pose-env/Scripts/activate

# 의존성 설치
pip install -r requirements.txt
```

## 사용 방법

1. 데이터 준비:
   ```bash
   python src/prepare_data.py
   ```

2. 모델 학습:
   ```bash
   python src/train.py configs/custom_config.py
   ```

3. 모델 테스트:
   ```bash
   python src/test_model.py
   ```

4. 결과 시각화:
   ```bash
   tensorboard --logdir results/
   ```

## 주요 기능

- 실시간 자세 분석
- 자세 교정 피드백
- 학습 진행 상황 모니터링
- 결과 시각화

## 의존성

- Python 3.8+
- PyTorch
- MMAction2
- OpenCV
- TensorBoard
