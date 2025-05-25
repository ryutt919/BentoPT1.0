# visualizer.py
import cv2 as cv
from config import COLORS, CONNECTIONS_BASIC

# 전체 선 굵기 조정
thickness = 1
border_thickness = thickness * 2

def draw_joints(frame, landmarks, joint_indices=()):
    for i in joint_indices:
        if i in landmarks:
            x, y = landmarks[i]
            color = (255,0,0) if i in (11,12) else COLORS.get(i, (255,255,255))
            cv.circle(frame, (x,y), 3, color, -1)

def draw_connections(frame, landmarks, connections):
    for a,b in connections:
        if a in landmarks and b in landmarks:
            s,e = landmarks[a], landmarks[b]
            cv.line(frame, s, e, (255,255,255), border_thickness)
            main_color = (255,0,0) if {a,b}=={11,12} else (0,0,255)
            cv.line(frame, s, e, main_color, thickness)

def draw_limbs(frame, landmarks, side):
    pairs = [(15,13),(13,11)] if side=='left' else [(16,14),(14,12)]
    draw_connections(frame, landmarks, pairs)

def draw_torso(frame, shoulder_center, spine, head_center):
    if shoulder_center:
        cv.circle(frame, shoulder_center, 6, (128,0,128), -1)
    if shoulder_center and spine:
        cv.line(frame, shoulder_center, spine, (255,255,255), border_thickness)
        cv.line(frame, shoulder_center, spine, (0,0,255), thickness)
    if shoulder_center and head_center:
        cv.line(frame, shoulder_center, head_center, (255,255,255), border_thickness)
        cv.line(frame, shoulder_center, head_center, (0,0,255), thickness)

def draw_shoulder_connection(frame, landmarks, is_frontal):
    """정/후면 시 파랑, 그 외 초록"""
    if 11 in landmarks and 12 in landmarks:
        s,e = landmarks[11], landmarks[12]
        color = (255,0,0) if is_frontal else (0,255,0)
        cv.line(frame, s, e, (255,255,255), border_thickness)
        cv.line(frame, s, e, color, thickness)
