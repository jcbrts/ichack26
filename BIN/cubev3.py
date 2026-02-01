import cv2
import mediapipe as mp
import math
import pygame
import os
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from mediapipe.tasks.python.core.base_options import BaseOptions

# --- CONFIGURATION ---
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
MODEL_PATH = "/Users/jacobroberts/Desktop/ICHack/ichack26/Jacob files/hand_landmarker.task"
CSV_PATH = "/Users/jacobroberts/Desktop/ICHack/ichack26/Jacob files/fusion_data.csv"

# SENSITIVITY
PAN_SPEED = 1.0
ZOOM_SPEED = 800.0
ROTATION_SENSITIVITY = 4.0 
SMOOTH_FACTOR = 0.15 

# --- PYGAME SETUP ---
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Fusion 360 Controller (Inverted Yaw)")
clock = pygame.time.Clock()

vertices = [
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
]
edges = [
    (0,1), (1,2), (2,3), (3,0),
    (4,5), (5,6), (6,7), (7,4),
    (0,4), (1,5), (2,6), (3,7)
]

# --- MATH HELPERS ---
def rotate_x(point, angle):
    x, y, z = point
    c, s = math.cos(angle), math.sin(angle)
    return [x, y * c - z * s, y * s + z * c]

def rotate_y(point, angle):
    x, y, z = point
    c, s = math.cos(angle), math.sin(angle)
    return [x * c + z * s, y, -x * s + z * c]

def rotate_z(point, angle):
    x, y, z = point
    c, s = math.cos(angle), math.sin(angle)
    return [x * c - y * s, x * s + y * c, z]

def project(point, scale, cx, cy):
    x, y, z = point
    f = scale / (4 + z) 
    return (int(x * f + cx), int(y * f + cy))

# --- GESTURE RECOGNITION ---

def is_finger_extended(landmarks, tip_idx, pip_idx, wrist):
    tip = landmarks[tip_idx]
    pip = landmarks[pip_idx]
    d_tip = math.hypot(tip.x - wrist.x, tip.y - wrist.y)
    d_pip = math.hypot(pip.x - wrist.x, pip.y - wrist.y)
    return d_tip > d_pip * 1.2

def is_peace_sign(hand_landmarks):
    wrist = hand_landmarks[0]
    index_open = is_finger_extended(hand_landmarks, 8, 6, wrist)
    middle_open = is_finger_extended(hand_landmarks, 12, 10, wrist)
    ring_closed = not is_finger_extended(hand_landmarks, 16, 14, wrist)
    pinky_closed = not is_finger_extended(hand_landmarks, 20, 18, wrist)
    return index_open and middle_open and ring_closed and pinky_closed

def is_fist(hand_landmarks):
    wrist = hand_landmarks[0]
    fingers_closed = 0
    for tip, pip in [(8,6), (12,10), (16,14), (20,18)]:
        if not is_finger_extended(hand_landmarks, tip, pip, wrist):
            fingers_closed += 1
    return fingers_closed >= 3

# --- MEDIAPIPE INIT ---
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    num_hands=2,
    running_mode=RunningMode.VIDEO,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)

# State Variables
cur_cx, cur_cy = WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2
cur_scale = 200.0
cur_pitch, cur_yaw, cur_roll = 0.0, 0.0, 0.0

tgt_cx, tgt_cy = cur_cx, cur_cy
tgt_scale = cur_scale
tgt_pitch, tgt_yaw, tgt_roll = 0.0, 0.0, 0.0

# Clutch Offsets
off_cx, off_cy = 0, 0
off_scale = 0
off_pitch, off_yaw, off_roll = 0.0, 0.0, 0.0
clutch_active = False

line_color = (0, 255, 0)

with HandLandmarker.create_from_options(options) as landmarker:
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = landmarker.detect_for_video(mp_image, int(pygame.time.get_ticks()))

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
        screen.blit(frame_surface, (0, 0))

        mode_text = "Status: Idle"
        
        if len(result.hand_landmarks) == 2:
            hands = sorted(result.hand_landmarks, key=lambda h: h[8].x)
            left_hand = hands[0]
            right_hand = hands[1]

            fist_l = is_fist(left_hand)
            fist_r = is_fist(right_hand)
            peace_l = is_peace_sign(left_hand)
            peace_r = is_peace_sign(right_hand)

            lx, ly = left_hand[8].x, left_hand[8].y
            rx, ry = right_hand[8].x, right_hand[8].y
            mx = (lx + rx) / 2
            my = (ly + ry) / 2
            dist = math.sqrt((rx - lx)**2 + (ry - ly)**2)
            angle = math.atan2(ry - ly, rx - lx)

            # --- CALCULATE RAW YAW (Inverted) ---
            # We multiply by -1 to invert the direction
            raw_yaw = -1 * (mx - 0.5) * ROTATION_SENSITIVITY * math.pi
            
            # Normal Pitch (Not inverted, unless you want that too)
            raw_pitch = (my - 0.5) * ROTATION_SENSITIVITY * math.pi

            if fist_l and fist_r:
                mode_text = "Status: CLUTCH LOCKED"
                line_color = (255, 0, 0)
                clutch_active = True
                
            else:
                if clutch_active:
                    # Reset Offsets
                    off_cx = cur_cx - (mx * WINDOW_WIDTH)
                    off_cy = cur_cy - (my * WINDOW_HEIGHT)
                    off_scale = cur_scale - (dist * ZOOM_SPEED)
                    off_roll = cur_roll - angle
                    off_yaw = cur_yaw - raw_yaw
                    off_pitch = cur_pitch - raw_pitch
                    clutch_active = False

                if peace_l and peace_r:
                    mode_text = "Status: 3D ROTATE (Peace Sign)"
                    line_color = (0, 0, 255)
                    
                    tgt_yaw = raw_yaw + off_yaw
                    tgt_pitch = raw_pitch + off_pitch
                    
                    # Prevent jumping in other modes
                    off_cx = cur_cx - (mx * WINDOW_WIDTH)
                    off_cy = cur_cy - (my * WINDOW_HEIGHT)
                    off_scale = cur_scale - (dist * ZOOM_SPEED)
                    
                else:
                    mode_text = "Status: PAN & ZOOM"
                    line_color = (0, 255, 0)
                    
                    tgt_cx = (mx * WINDOW_WIDTH) + off_cx
                    tgt_cy = (my * WINDOW_HEIGHT) + off_cy
                    tgt_scale = (dist * ZOOM_SPEED) + off_scale
                    tgt_roll = angle + off_roll
                    
                    # Prevent jumping in rotation
                    off_yaw = cur_yaw - raw_yaw
                    off_pitch = cur_pitch - raw_pitch

            p1 = (int(lx * WINDOW_WIDTH), int(ly * WINDOW_HEIGHT))
            p2 = (int(rx * WINDOW_WIDTH), int(ry * WINDOW_HEIGHT))
            pygame.draw.line(screen, line_color, p1, p2, 5)

        # 4. SMOOTHING
        cur_cx += (tgt_cx - cur_cx) * SMOOTH_FACTOR
        cur_cy += (tgt_cy - cur_cy) * SMOOTH_FACTOR
        cur_scale += (tgt_scale - cur_scale) * SMOOTH_FACTOR
        cur_pitch += (tgt_pitch - cur_pitch) * SMOOTH_FACTOR
        cur_yaw += (tgt_yaw - cur_yaw) * SMOOTH_FACTOR
        cur_roll += (tgt_roll - cur_roll) * SMOOTH_FACTOR

        # 5. CSV WRITING
        norm_pan_x = (cur_cx - WINDOW_WIDTH / 2) / (WINDOW_WIDTH / 2)
        norm_pan_y = -1 * (cur_cy - WINDOW_HEIGHT / 2) / (WINDOW_HEIGHT / 2)
        norm_zoom = cur_scale / 200.0

        try:
            with open(CSV_PATH, "w") as f:
                f.write("pan_x,pan_y,zoom,rot_x,rot_y,rot_z\n")
                f.write(f"{norm_pan_x:.4f},{norm_pan_y:.4f},{norm_zoom:.4f},{cur_pitch:.4f},{cur_yaw:.4f},{cur_roll:.4f}\n")
        except:
            pass

        # 6. RENDER
        transformed_points = []
        for v in vertices:
            p = rotate_x(v, cur_pitch)
            p = rotate_y(p, cur_yaw)
            p = rotate_z(p, cur_roll)
            transformed_points.append(p)

        for edge in edges:
            start_2d = project(transformed_points[edge[0]], cur_scale, cur_cx, cur_cy)
            end_2d = project(transformed_points[edge[1]], cur_scale, cur_cx, cur_cy)
            pygame.draw.line(screen, (255, 255, 255), start_2d, end_2d, 4)

        font = pygame.font.SysFont("Arial", 24)
        tsurf = font.render(mode_text, True, line_color)
        screen.blit(tsurf, (20, 20))
        
        instr = font.render("Fists = Clutch | Peace = Spin | Open = Move", True, (200, 200, 200))
        screen.blit(instr, (20, 50))

        pygame.display.flip()
        clock.tick(60)

cap.release()
pygame.quit()