# analyzer.py
# 운동별 분석 파이프라인: utils.get_angle, counter.RepCounter 통합

from utils import get_angle
from counter import RepCounter
from config import EXERCISE_PARAMS, HEAD_LANDMARKS

# 모듈 레벨 카운터 저장소, 운동별로 하나씩 유지
_counters = {}

def analyze_exercise(name, landmarks, now=None):
    """
    주어진 운동 이름(name)과 랜드마크를 분석하여
    rep 카운트 증감, 템포 메시지, 자세 피드백 메시지를 반환합니다.

    Args:
        name (str): 'pushup', 'pullup', 'squat', 'lunge', 'biceps_curl'
        landmarks (dict): {landmark_index: (x, y)}
        now (float, optional): 타임스탬프
    Returns:
        tuple: (count_inc: int, tempo_msg: Optional[str], angle_msg: Optional[str])
    """
    # 카운터 초기화
    if name not in _counters:
        params = EXERCISE_PARAMS[name]
        _counters[name] = RepCounter(params['angle_thresholds'], params['tempo'])
    counter = _counters[name]

    # 운동별 주요 관절 인덱스 지정
    if name == 'pushup':
        # 팔꿈치 중심(p2), 어깨(p1), 손목(p3)
        idx_p1, idx_p2, idx_p3 = 11, 13, 15
    elif name == 'pullup':
        idx_p1, idx_p2, idx_p3 = 15, 13, 11
    elif name == 'squat':
        idx_p1, idx_p2, idx_p3 = 23, 24, 26  # 예시: 골반-무릎-발목
    elif name == 'lunge':
        idx_p1, idx_p2, idx_p3 = 23, 24, 26
    elif name == 'biceps_curl':
        idx_p1, idx_p2, idx_p3 = 15, 13, 11
    else:
        raise ValueError(f"Unsupported exercise: {name}")

    # 필요한 랜드마크가 존재하는지 체크
    try:
        p1 = landmarks[idx_p1]
        p2 = landmarks[idx_p2]
        p3 = landmarks[idx_p3]
    except KeyError:
        return 0, None, None

    # 각도 계산
    angle = get_angle(p1, p2, p3)

    # rep 카운트 및 템포
    count_inc, tempo_msg = counter.update(angle, now)

    # 자세 피드백
    thr = counter.angle_up if counter.state == 'up' else counter.angle_down
    angle_msg = None
    if counter.state == 'up' and angle < thr:
        angle_msg = 'angle_bad'
    elif counter.state == 'down' and angle > thr:
        angle_msg = 'angle_bad'
    else:
        angle_msg = 'angle_good'

    return count_inc, tempo_msg, angle_msg
