BTPT_1.0\n
도시락처럼 들고다니는 PT선생님
----

데이터 수집 및 전처리 과정
1. 영상에서 사람 검출 및 키포인트 추출
사용 모델: Ultralytics YOLOv8 기반의 yolov8x.pt (객체 검출) 및 yolov8x-pose.pt (포즈 추정) 모델을 사용

작업 내용:

입력 영상 프레임마다 사람을 검출하고, 검출된 사람에 대해 17개 주요 관절 키포인트(x, y 좌표 + 신뢰도) 추출

여러 사람 검출 시 해당 영상은 제외하고, 한 명만 검출된 프레임만 처리

일정 간격(frame_interval)으로 프레임을 추출해 키포인트 데이터 .npz 파일로 저장 (프레임 번호, 타임스탬프, 키포인트 배열 포함)

2. 키포인트 보정 및 필터링
필터링 기법:

메디안 필터: 이상치 제거

가우시안 필터: 부드러운 스무딩

1D 칼만 필터: 시계열 노이즈 감소 및 예측 보정

결과: 원본 키포인트 데이터에서 신뢰도가 낮거나 노이즈가 많은 좌표를 보완해 안정적인 관절 위치 좌표를 생성

3. 좌표 정규화
원본 영상 크기(width, height)를 기준으로 키포인트의 x, y 좌표를 [0, 1] 범위로 정규화하여 크기와 해상도 차이에 무관한 데이터로 변환

-----

라벨링 및 데이터 준비
4. 수동 라벨링
.npz 파일과 원본 영상을 연동하여 각 프레임별 운동 자세 상태를 라벨링

라벨링 키맵에 따라 '좋은 자세', '팔꿈치 각도 문제', '무릎 위치 불량' 등 다양한 피드백 태그 부착 가능

라벨링 결과는 CSV 파일로 저장하여 후속 학습용 레이블로 활용

5. MMACTION2 데이터셋 포맷 변환 및 시퀀스 생성
키포인트 데이터를 COCO 포맷에 맞게 재구성 (17개 관절)

슬라이딩 윈도우(예: 길이 8, 겹침 4) 방식으로 시퀀스 분할

각 시퀀스에 라벨 부착 (라벨 데이터 존재 시)

학습 및 검증용 데이터셋으로 분할 후 Pickle 포맷으로 저장

모델 학습 및 추론
6. 학습 환경 및 실행
프레임워크: MMAction2 기반 학습 파이프라인 사용

모델: ST-GCN(Spatial Temporal Graph Convolutional Network) 구조 등 시계열 관절 데이터 학습

학습 스크립트: train.py, run_train.py에서 config 파일 지정 및 학습 옵션 설정 (랜덤 시드 고정, AMP 지원 등)

평가 지표: 정확도, 정밀도, 재현율, F1 점수 등

학습 완료 후 best checkpoint 저장

7. 준지도 학습 (Pseudo Labeling)
라벨이 없는 데이터셋에 대해 학습된 모델로 예측 수행

신뢰도가 높은 예측 결과만 선별해 가짜 라벨(pseudo-label)로 부착

pseudo-label 데이터를 기존 학습 데이터셋에 추가해 재학습하여 성능 향상 도모

시각화 및 피드백 기능
8. 키포인트 및 스켈레톤 시각화
OpenCV를 활용해 원본 영상 위에 키포인트별 원(왼쪽 관절은 주황색, 오른쪽 관절은 초록색, 중앙 관절은 흰색)과 관절 연결선 그리기

발목과 무릎 간 보조선 등 운동 분석에 필요한 참조선 추가

영상 재생과 함께 키포인트 상태 실시간 확인 가능

9. 인터랙티브 라벨링 UI
키보드 입력으로 각 프레임별 피드백 라벨 토글 및 저장 가능

라벨 저장, 삭제, 이전/다음 프레임 이동 기능 제공

주요 기술 스택 및 라이브러리
Python 3.x

OpenCV (영상 처리 및 시각화)

Ultralytics YOLOv8 (사람 검출 및 포즈 추정)

NumPy, SciPy (수치 연산 및 필터링)

pandas (라벨 CSV 처리)

PyTorch 및 MMAction2 (모션 인식 모델 학습 및 추론)

tqdm (진행 상황 표시)

Scikit-learn (데이터 분할 및 평가 지표)




예시
![image](https://github.com/user-attachments/assets/558d1a7e-1b03-460d-9a75-9d626525b801)
![image](https://github.com/user-attachments/assets/fc884bbe-7b65-4745-8b33-d6f1a2005cf1)

![image](https://github.com/user-attachments/assets/ae76982d-cff3-4648-87aa-19268366e8e7)

런지
그래
![image](https://github.com/user-attachments/assets/45666544-166d-432a-8bb2-597ed9636b7c)
탑3
![image](https://github.com/user-attachments/assets/c5d4f81e-c834-4ebb-8c96-014f4fcf7302)

풀어ㅓ1
![image](https://github.com/user-attachments/assets/9543df6e-479c-485a-b9e3-11194ca070b6)
3
![image](https://github.com/user-attachments/assets/9e661e8f-f3be-47c4-b6fb-fade493d7a2c)
