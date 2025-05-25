# exercise.py
# 운동 클래스를 통해 분석 파이프라인을 추상화한 모듈

from analyzer import analyze_exercise
from config import EXERCISE_PARAMS, FEEDBACK_MESSAGES

class ExerciseBase:
    """
    각 운동을 추상화하는 클래스. 이름(name)으로 운동 종류를 지정하고,
    update() 호출 시 분석 결과로 횟수, 피드백 메시지 리스트를 반환합니다.

    Args:
        name (str): 'pushup', 'pullup', 'squat', 'lunge', 'biceps_curl'
    """
    def __init__(self, name):
        if name not in EXERCISE_PARAMS:
            raise ValueError(f"Unsupported exercise: {name}")
        self.name = name
        self.params = EXERCISE_PARAMS[name]
        self.total_count = 0

    def update(self, landmarks, now=None):
        """
        주어진 landmarks와 timestamp(now)로 운동 상태를 분석합니다.

        Returns:
            total_count (int): 누적 rep 수
            feedback_list (List[str]): count, tempo, angle 피드백 메시지 리스트
        """
        # 분석 모듈 호출
        count_inc, tempo_code, angle_code = analyze_exercise(self.name, landmarks, now)
        # 누적 횟수 업데이트
        self.total_count += count_inc

        feedback_list = []
        # 1) 횟수 메시지
        count_msg = FEEDBACK_MESSAGES['count'].format(
            exercise=self.name,
            count=self.total_count
        )
        feedback_list.append(count_msg)

        # 2) 템포 메시지 (코드 기반 템플릿 적용)
        if tempo_code:
            # 'tempo_too_fast', 'tempo_too_slow' 중 하나
            key = tempo_code.split()[0]
            tmpl = FEEDBACK_MESSAGES.get(key)
            if tmpl:
                # 템플릿에는 min_time, max_time 자리 있음
                tempo_msg = tmpl.format(
                    min_time=self.params['tempo']['min_time'],
                    max_time=self.params['tempo']['max_time']
                )
                feedback_list.append(tempo_msg)
            else :
                feedback_list.append("템포")

        # 3) 각도 피드백
        if angle_code:
            angle_msg = FEEDBACK_MESSAGES.get(angle_code)
            if angle_msg:
                feedback_list.append(angle_msg)

        return self.total_count, feedback_list
