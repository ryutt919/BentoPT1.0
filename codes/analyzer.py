# analyzer.py
from utils import calculate_angle

def evaluate_pushup(landmarks):
    # 왼팔 관절이 모두 존재할 경우에만 평가 수행
    required = [11, 13, 15]
    if not all(k in landmarks for k in required):
        return "왼팔 관절이 감지되지 않았습니다."

    angle = calculate_angle(landmarks[11], landmarks[13], landmarks[15])
    
    if 80 < angle < 100:
        return "정확한 자세입니다."
    else:
        return "팔을 더 곧게 펴세요."
