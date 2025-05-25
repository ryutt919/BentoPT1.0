import time
from utils import format_time

class RepCounter:
    """
    반복 운동(rep) 횟수 카운팅 및 템포 측정을 위한 상태 머신.
    Full cycle(down→up→down) 완료 시 카운트.
    """
    def __init__(self, angle_thresholds, tempo_thresholds):
        # 임계치 설정
        self.angle_up = angle_thresholds['up']
        self.angle_down = angle_thresholds['down']
        self.min_time = tempo_thresholds['min_time']
        self.max_time = tempo_thresholds['max_time']

        # 상태 초기화
        self.state = 'init'      # 'init', 'up', 'down'
        self.count = 0
        self.last_transition_time = None

    def update(self, angle, now=None):
        """
        현재 각도를 받아 상태 전환 및 rep 카운팅, 템포 메시지 생성.

        Returns:
            count_inc (int): 이번 호출로 증가된 rep 수 (0 또는 1)
            tempo_msg (str or None): 템포 관련 메시지
        """
        if now is None:
            now = time.time()

        count_inc = 0
        tempo_msg = None

        # 초기 상태 설정
        if self.state == 'init':
            if angle <= self.angle_down:
                self.state = 'down'
            elif angle >= self.angle_up:
                self.state = 'up'
            self.last_transition_time = now
            return 0, None

        # down 상태: 상단 도달 대기
        if self.state == 'down':
            if angle >= self.angle_up:
                self.state = 'up'
                # 상단 도달 시 시간 기록
                self.last_transition_time = now

        # up 상태: 하단 복귀시 카운트
        elif self.state == 'up':
            if angle <= self.angle_down:
                self.state = 'down'
                # Full cycle complete
                self.count += 1
                count_inc = 1
                # 템포 측정
                delta = now - (self.last_transition_time or now)
                tempo_msg = self._evaluate_tempo(delta)
                self.last_transition_time = now

        return count_inc, tempo_msg

    def _evaluate_tempo(self, delta):
        """
        시간 차(delta)에 따른 템포 메시지 반환.
        """
        if delta < self.min_time:
            return f'tempo_too_fast ({format_time(delta)})'
        if delta > self.max_time:
            return f'tempo_too_slow ({format_time(delta)})'
        return None
