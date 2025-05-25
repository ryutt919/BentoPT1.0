# feedback.py
# 화면에 피드백 텍스트를 그려주는 모듈

import cv2 as cv

def draw_feedback(frame, message, position=(30, 30)):
    """
    frame: 그릴 프레임,
    message: 출력할 문자열,
    position: 텍스트 시작 좌표 (x, y)
    """
    # cv.putText(frame,
    #            message,
    #            position,
    #            cv.FONT_HERSHEY_SIMPLEX,
    #            1.0,         # 글자 크기
    #            (0, 0, 255), # 빨간색
    #            2)           # 두께
