# 🍱 Bento PT 1.0 🏋️‍♂️

##  프로젝트 개요  
OpenCV와 Python 기반의 이 시스템은 사용자가 짧은 운동 영상과 운동 종류를 입력하면,  
**실시간 영상 재생**, **관절 키포인트 시각화**, **보조선 출력**을 통해 **운동 자세를 분석하고 피드백**을 제공합니다.

---
## 개발 동기 및 배경
이 프로젝트는 단순히 돌아가는 시스템을 만드는 것이 목표가 아니었습니다.
나는 수업 시간에 배운 내용들을 실제 문제 해결에 적용하고, 기존에 있는 모델이나 자동화된 라이브러리에만 의존하지 않고,
가능한 한 모든 구성요소를 직접 설계하고 학습시키며 커스터마이징해보고 싶었습니다.

그래서 선택한 방법은 쉽지 않았습니다.
수많은 포즈 추정 오픈소스와 사전학습된 모델들이 넘쳐나는 시대에,
나는 직접 YOLO 기반 키포인트 추출 → 필터링 → 정규화 → 라벨링 → ST-GCN 학습까지 모든 파이프라인을 커스텀 구축했습니다.

모델 학습을 위해 Kinetics-400 기반의 운동 영상을 수집하고
라벨링부터 데이터 구조 변환, 시퀀스 구성까지 하나하나 직접 설계하며
혼자서 MMACTION2의 광대한 코드베이스와 문서 속을 끝없이 헤매고,
수십 번의 시행착오와 수많은 삽질 끝에 결국 모델을 학습시켜냈습니다.

이 프로젝트는 단지 '잘 작동하는 프로그램'이 아니라
"하나의 구조를 이해하고 스스로 빌드업한 경험"이며,
수업에서 배운 이론이 실제로 어떻게 구현되고 확장될 수 있는지를 증명한 개인적 증거이자 기록입니다.
---

##  주요 기능 및 처리 흐름

### 1.  사람 검출 및 키포인트 추출
- **사용 모델**: `YOLOv8x` (객체 감지), `YOLOv8x-pose` (포즈 추정)
- **작업 내용**:
  - 프레임마다 사람 감지 → 17개 관절 키포인트 추출 `(x, y, confidence)`
  - 학습 데이터 질 향상을 위해 다중 인물 영상은 자동 제외
  - 결과는 `.npz` 파일로 저장 (`frame`, `timestamp`, `keypoints` 포함)

---

### 2.  키포인트 보정 및 필터링
- **필터링 기법**:
  -  메디안 필터: 이상치 제거  
  -  가우시안 필터: 부드러운 스무딩  
  -  1D 칼만 필터: 시계열 노이즈 감소 및 예측 보정  
- **정규화 처리**:  
  - 프레임 해상도에 따라 키포인트 좌표를 `[0, 1]` 범위로 정규화

---

### 3.  수동 라벨링 도구
- `.npz` + 영상 동기화 → 영상 별로 일부 프레임에 직접 라벨링
- 키보드 단축키로 `good`, `elbow_not_vertical`, `knee_over_toes` 등 피드백 라벨 부여
- 라벨은 `.csv` 파일로 저장되어 학습용 라벨로 사용됨

---

### 4.  데이터셋 준비 (MMAction2 호환)
- **COCO 포맷**으로 키포인트 변환  
- **슬라이딩 윈도우** 방식으로 시퀀스 생성 (예: `window_size=8`, `stride=4`)  
- **라벨링된/되지 않은 시퀀스 분리**  
- MMACTION2 학습용 `train.pkl`, `val.pkl`, `unlabeled.pkl` 등으로 저장

---

### 5. 🧠 모델 학습
- **프레임워크**: [MMAction2](https://github.com/open-mmlab/mmaction2)  
- **모델**: ST-GCN (Spatial Temporal Graph Convolutional Network)
- **특징**:
  운동 자세 분석은 단순한 프레임 간 변화만으로는 충분하지 않으며,
  사람의 **관절 간 연결 구조(관절-관절 관계)**와 **시간 축 움직임(동작 흐름)**을 동시에 반영해야 합니다.

  기존 CNN은 픽셀 격자 기반이기 때문에, 관절 간의 불규칙하고 비유클리드적인 연결 관계를 표현하기 어렵습니다.
  따라서 본 프로젝트는 다음의 이유로 ST-GCN (Spatial-Temporal Graph Convolutional Network) 모델을 채택했습니다
  
- **구조**
  구성 요소	설명
  Spatial Graph	각 프레임의 관절을 그래프 노드로 구성하고, 관절 간 해부학적 연결에 따라 엣지를 구성
  Temporal Graph	동일 관절이 시간 축을 따라 이어지도록 연결 → 움직임의 연속성 반영
  Graph Convolution	CNN 대신 Graph Convolution을 통해 관절 간 구조적인 관계 학습
  Joint-Level Attention	특정 관절의 움직임이 자세 평가에 얼마나 중요한지를 학습함


---

### 6.  준지도 학습 (Pseudo Labeling)
- **라벨 없는 시퀀스**에 대해 학습된 모델로 예측 수행
- **신뢰도 기준 이상(≥ 0.8)**인 예측 결과만 가짜 라벨로 추가
- Pseudo-label 데이터까지 포함해 학습 데이터셋 확장 및 성능 향상
- 
  **학습 결과**
- 학습 데이터의 질적 차이로 인해 풀업이 더 좋은 성능을 보여줌
- 런지의 경우 window size와 stride와 같은 파라미터를 조정하여 성능 향상을 보여줌<br>
  <br>
 
- 런지
- ![image](https://github.com/user-attachments/assets/45666544-166d-432a-8bb2-597ed9636b7c)
- ![image](https://github.com/user-attachments/assets/c5d4f81e-c834-4ebb-8c96-014f4fcf7302)

- 풀업
- ![image](https://github.com/user-attachments/assets/9543df6e-479c-485a-b9e3-11194ca070b6)
- ![image](https://github.com/user-attachments/assets/9e661e8f-f3be-47c4-b6fb-fade493d7a2c)

---

### 7.  키포인트 시각화 및 피드백 제공
- GUI를 통한 편리한 사용 제공
- ![image](https://github.com/user-attachments/assets/fc884bbe-7b65-4745-8b33-d6f1a2005cf1)

- **OpenCV** 기반 실시간 키포인트/스켈레톤 오버레이
- 관절 연결선 + 발목-무릎 보조선 포함
- 실시간 피드백 제공
- ![image](https://github.com/user-attachments/assets/ae76982d-cff3-4648-87aa-19268366e8e7)


---

##  사용 기술 스택

| 도구 | 설명 |
|------|------|
| Python 3.x | 전체 프로젝트 구현 |
| OpenCV | 영상 처리 및 키포인트 시각화 |
| YOLOv8 | 사람 검출 및 관절 키포인트 추출 |
| NumPy / SciPy | 필터링 및 수치 연산 |
| pandas | 라벨 CSV 처리 |
| PyTorch / MMAction2 | ST-GCN 모델 학습 및 추론 |
| tqdm | 진행 상황 표시 |
| scikit-learn | 데이터 분할 및 평가 지표 계산 |

---



