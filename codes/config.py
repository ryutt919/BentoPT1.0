# config.py
# 프로젝트 전역에서 사용하는 설정값들을 모아둔 파일

# 지원 운동 목록
EXERCISE_LIST = [
    "pullup",
    "pushup",
    "squat",
    "lunge",
    "biceps_curl"
]
# MediaPipe 관절 인덱스 (https://developers.google.com/mediapipe/solutions/vision/pose_landmarker#output 참고)
JOINT_INDICES = [
    11, 12,  # 왼/오른 어깨
    13, 14,  # 왼/오른 팔꿈치
    15, 16,  # 왼/오른 손목
    23, 24,  # 왼/오른 골반
     0       # 코 (머리 전면)
]

# 머리 위치 추정용 랜드마크
HEAD_LANDMARKS = {
    'left_ear': 7,
    'right_ear': 8,
    'nose': 0
}

# 관절 간 기본 연결 쌍 (어깨↔팔꿈치↔손목, 어깨 좌우)
CONNECTIONS_BASIC = [
    (15, 13), (13, 11),  # 왼팔
    (16, 14), (14, 12),  # 오른팔
    (11, 12)             # 어깨 좌우
]

# 관절 점 및 귀 표시 색상 (B, G, R)
COLORS = {
    11: (255,   0,   0),  # 왼 어깨
    12: (  0, 255,   0),  # 오른 어깨
    13: (  0,   0, 255),  # 왼 팔꿈치
    14: (255, 255,   0),  # 오른 팔꿈치
    15: (  0, 255, 255),  # 왼 손목
    16: (255,   0, 255),  # 오른 손목
     0: (255, 255, 255),  # 코
     7: (128,   0, 128),  # 왼 귀
     8: (  0, 128, 128)   # 오른 귀
}

# 입력 키 바인딩
PAUSE_KEY  = ' '
FLIP_KEY   = 'f'
RECORD_KEY = 'r'

# 시각화 옵션
SCALE     = 1.5
THICKNESS = 2

# 기본 피드백 메시지 템플릿
FEEDBACK_MESSAGES = {
    'count': "{exercise}: {count} reps",
    'tempo_too_fast': "천천히 하세요 (min_time: {min_time}s)",
    'tempo_too_slow': "속도를 높이세요 (max_time: {max_time}s)",
    'angle_good': "좋은 자세입니다",
    'angle_bad': "자세를 교정하세요"
}

# 운동별 파라미터: 각도 임계치 및 템포 기준 (초 단위)
EXERCISE_PARAMS = {
    'pushup': {
        'angle_thresholds': {'up': 160, 'down': 90},
        'tempo': {'min_time': 1.0, 'max_time': 3.0}
    },
    'pullup': {
        'angle_thresholds': {'up': 40, 'down': 160},
        'tempo': {'min_time': 1.0, 'max_time': 4.0}
    },
    'squat': {
        'angle_thresholds': {'up': 170, 'down': 90},
        'tempo': {'min_time': 1.0, 'max_time': 3.0}
    },
    'lunge': {
        'angle_thresholds': {'up': 180, 'down': 90},
        'tempo': {'min_time': 1.0, 'max_time': 3.0}
    },
    'biceps_curl': {
        'angle_thresholds': {'up': 40, 'down': 160},
        'tempo': {'min_time': 1.0, 'max_time': 3.0}
    }
}
