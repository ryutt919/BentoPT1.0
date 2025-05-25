# visualizer.py
# 랜드마크, 연결선, 중심점, 피드백 텍스트 등을 그리는 모듈

import cv2 as cv
import numpy as np
from config import CONNECTIONS_BASIC, COLORS
from PIL import Image, ImageDraw, ImageFont

# 중심점 이름별 기본 색상 매핑
CENTER_COLORS = {
    'head': COLORS.get(0, (255, 255, 255)),
    'shoulder': COLORS.get(11, (0, 255, 0)),
    'hip': COLORS.get(23, (0, 255, 255)),
    'spine': COLORS.get(23, (0, 255, 255))
}

# 한국어 지원 가능한 폰트 경로 리스트
FONT_PATHS = [
    'C:/Windows/Fonts/malgun.ttf',
    '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc'
]


def draw_landmarks(frame, landmarks, scale=0.5, thickness=1):
    """
    랜드마크 점을 프레임에 그립니다. 점 크기를 줄여 배경에 덜 방해하도록 함.
    """
    # radius = max(2, int(2 * scale))  # 점 크기 축소
    # for idx, (x, y) in landmarks.items():
    #     cv.circle(frame, (x, y), radius, COLORS.get(idx, (0, 0, 255)), -1)


def draw_connections(frame, landmarks, connections=CONNECTIONS_BASIC, thickness=1):
    """
    검정 테두리 + 빨간 본선으로 연결선을 그려 시인성 강화.
    """
    outer_thickness = thickness*3
    inner_thickness = thickness
    for idx1, idx2 in connections:
        if idx1 in landmarks and idx2 in landmarks:
            pt1 = landmarks[idx1]
            pt2 = landmarks[idx2]
            # 검정 테두리
            cv.line(frame, pt1, pt2, (0, 0, 0), outer_thickness)
            # 빨간 본선
            cv.line(frame, pt1, pt2, (0, 0, 255), inner_thickness)


def draw_centers(frame, centers, thickness=1):
    """
    주요 중심점(머리, 어깨, 골반, 척추)을 그리고,
    head↔spine은 점 대신 선으로 연결합니다.
    """
    outer_thickness = thickness*3
    inner_thickness = thickness

    # 머리(head)와 척추(spine)를 선으로 연결
    if centers.get('head') and centers.get('spine'):
        pt1 = centers['head']
        pt2 = centers['spine']
        # 테두리 선 (검정, 두께 3×scale)
        cv.line(frame, pt1, pt2, (0, 0, 0), outer_thickness)
        # 본선 (빨강, 두께 1×scale)
        cv.line(frame, pt1, pt2, (255, 0, 0), inner_thickness)


def draw_feedback(frame, feedback_list, scale=1.0, thickness=2):
    """
    화면에 한글 및 영문 피드백 메시지 리스트를 출력합니다.
    OpenCV에서 한글이 깨지는 문제를 해결하기 위해 Pillow 활용.
    """
    # OpenCV BGR -> PIL RGB
    img_pil = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # 한글 지원 폰트 로드
    font = None
    for path in FONT_PATHS:
        try:
            font = ImageFont.truetype(path, int(1 * scale))
            break
        except IOError:
            continue
    if font is None:
        font = ImageFont.load_default()

    x0, y0 = int(10 * scale), int(10 * scale)
    line_height = int(20 * scale)
    # 피드백 문자열 렌더링
    for i, text in enumerate(feedback_list):
        y = y0 + i * line_height
        # Pillow 8+ 호환:
        try:
            text_w, text_h = font.getsize(text)
        except AttributeError:
            # 최신 Pillow: textbbox 사용
            bbox = draw.textbbox((x0, y), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]        # 백그라운드 박스
        draw.rectangle([x0 - 5, y - 5, x0 + text_w + 5, y + text_h + 5], fill=(255, 255, 255))
        # 텍스트
        draw.text((x0, y), text, font=font, fill=(0, 0, 0))

    # PIL RGB -> OpenCV BGR
    frame[:, :] = cv.cvtColor(np.array(img_pil), cv.COLOR_RGB2BGR)
