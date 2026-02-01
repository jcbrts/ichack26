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

# --- TUNING SETTINGS ---
PAN_SPEED = 1.0
ZOOM_SPEED = 800.0

# REDUCED SENSITIVITY (Was 4.0)
ROTATION_SENSITIVITY = 1.5 

# SMOOTHING FACTORS (0.01 = Very Slow/Smooth, 0.9 = Fast/Jittery)
PAN_ZOOM_SMOOTHING = 0.15 
ROTATION_SMOOTHING = 0.08  # New extra smoothing for rotation

# --- PYGAME SETUP ---
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Fusion 360 Controller (Smoothed Tumble)")
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

# --- MATRIX MATH HELPERS ---
def mat_mul(A, B):
    C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for i in range(3):
        for j in range(3):
            C[i][j] = A[i][0]*B[0][j] + A[i][1]*B[1][j] + A[i][2]*B[2][j]
    return C

def mat_vec_mul(M, v):
    x, y, z = v
    nx = M[0][0]*x + M[0][1]*y + M[0][2]*z
    ny = M[1][0]*x + M[1][1]*y + M[1][2]*z
    nz = M[2][0]*x + M[2][1]*y + M[2][2]*z
    return [nx, ny, nz]

def get_rot_x(angle):
    c, s = math.cos(angle), math.sin(angle)
    return [[1, 0, 0], [0, c, -s], [0, s, c]]

def get_rot_y(angle):
    c, s = math.cos(angle), math.sin(angle)
    return [[c, 0, s], [0, 1, 0], [-s, 0, c]]

def get_rot_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return [[c, -s, 0], [s, c, 0], [0, 0, 1]]

def matrix_to_euler(R):
    sy = math.sqrt(R[0][0] * R[0][0] + R[1][0] * R[1][0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2][1], R[2][2])
        y = math.atan2(-R[2][0], sy)
        z = math.atan2(R[1][0], R[0][0])
    else:
        x = math.atan2(-R[1][2], R[1][1])
        y = math.atan2(-R[2][0], sy)
        z = 0
    return x, y, z

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

# --- STATE VARIABLES ---
cur_rotation_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
cur_cx, cur_cy = WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2
cur_scale = 200.0
tgt_cx, tgt_cy = cur_cx, cur_cy
tgt_scale = cur_scale

prev_mx, prev_my = None, None
prev_angle = None

# Variables for rotation smoothing
smooth_rot_x = 0.0
smooth_rot_y = 0.0
smooth_rot_z = 0.0

# Offsets
off_cx, off_cy = 0, 0
off_scale = 0
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
        
        # Reset smoothed rotation targets to 0 every frame
        # We only want to rotate if there is ACTIVE input
        target_rot_x = 0.0
        target_rot_y = 0.0
        target_rot_z = 0.0

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

            if prev_mx is None:
                prev_mx, prev_my, prev_angle = mx, my, angle

            if fist_l and fist_r:
                mode_text = "Status: CLUTCH LOCKED"
                line_color = (255, 0, 0)
                clutch_active = True
                prev_mx, prev_my, prev_angle = mx, my, angle
                
            else:
                if clutch_active:
                    off_cx = cur_cx - (mx * WINDOW_WIDTH)
                    off_cy = cur_cy - (my * WINDOW_HEIGHT)
                    off_scale = cur_scale - (dist * ZOOM_SPEED)
                    prev_mx, prev_my, prev_angle = mx, my, angle
                    clutch_active = False

                if peace_l and peace_r:
                    # --- GLOBAL TUMBLE (Peace) ---
                    mode_text = "Status: 3D TUMBLE (Peace Sign)"
                    line_color = (0, 0, 255)
                    
                    # Calculate raw Delta
                    raw_delta_x = (mx - prev_mx) * ROTATION_SENSITIVITY * math.pi
                    raw_delta_y = (my - prev_my) * ROTATION_SENSITIVITY * math.pi
                    
                    # Assign to targets (Inverted X for Yaw)
                    target_rot_y = -raw_delta_x 
                    target_rot_x = raw_delta_y
                        
                    # Sync offsets so pan/zoom doesn't jump
                    off_cx = cur_cx - (mx * WINDOW_WIDTH)
                    off_cy = cur_cy - (my * WINDOW_HEIGHT)
                    off_scale = cur_scale - (dist * ZOOM_SPEED)

                else:
                    # --- PAN, ZOOM, & ROLL (Open) ---
                    mode_text = "Status: PAN, ZOOM & ROLL"
                    line_color = (0, 255, 0)
                    
                    # 1. Pan & Zoom Targets
                    tgt_cx = (mx * WINDOW_WIDTH) + off_cx
                    tgt_cy = (my * WINDOW_HEIGHT) + off_cy
                    tgt_scale = (dist * ZOOM_SPEED) + off_scale
                    
                    # 2. Roll Target
                    delta_roll = angle - prev_angle
                    if delta_roll > math.pi: delta_roll -= 2*math.pi
                    if delta_roll < -math.pi: delta_roll += 2*math.pi
                    
                    if abs(delta_roll) > 0.002:
                        target_rot_z = delta_roll

            prev_mx, prev_my, prev_angle = mx, my, angle

            p1 = (int(lx * WINDOW_WIDTH), int(ly * WINDOW_HEIGHT))
            p2 = (int(rx * WINDOW_WIDTH), int(ry * WINDOW_HEIGHT))
            pygame.draw.line(screen, line_color, p1, p2, 5)

        # --- ROTATION SMOOTHING LOOP ---
        # Smoothly interpolate current rotation speed towards the target speed
        smooth_rot_x += (target_rot_x - smooth_rot_x) * ROTATION_SMOOTHING
        smooth_rot_y += (target_rot_y - smooth_rot_y) * ROTATION_SMOOTHING
        smooth_rot_z += (target_rot_z - smooth_rot_z) * ROTATION_SMOOTHING

        # Apply smoothed rotations to the Matrix
        if abs(smooth_rot_y) > 0.0001:
            cur_rotation_matrix = mat_mul(get_rot_y(smooth_rot_y), cur_rotation_matrix)
        if abs(smooth_rot_x) > 0.0001:
            cur_rotation_matrix = mat_mul(get_rot_x(smooth_rot_x), cur_rotation_matrix)
        if abs(smooth_rot_z) > 0.0001:
            cur_rotation_matrix = mat_mul(get_rot_z(smooth_rot_z), cur_rotation_matrix)

        # --- PAN/ZOOM SMOOTHING ---
        cur_cx += (tgt_cx - cur_cx) * PAN_ZOOM_SMOOTHING
        cur_cy += (tgt_cy - cur_cy) * PAN_ZOOM_SMOOTHING
        cur_scale += (tgt_scale - cur_scale) * PAN_ZOOM_SMOOTHING

        # 5. CSV WRITING
        rot_x_out, rot_y_out, rot_z_out = matrix_to_euler(cur_rotation_matrix)
        norm_pan_x = (cur_cx - WINDOW_WIDTH / 2) / (WINDOW_WIDTH / 2)
        norm_pan_y = -1 * (cur_cy - WINDOW_HEIGHT / 2) / (WINDOW_HEIGHT / 2)
        norm_zoom = cur_scale / 200.0

        try:
            with open(CSV_PATH, "w") as f:
                f.write("pan_x,pan_y,zoom,rot_x,rot_y,rot_z\n")
                f.write(f"{norm_pan_x:.4f},{norm_pan_y:.4f},{norm_zoom:.4f},{rot_x_out:.4f},{rot_y_out:.4f},{rot_z_out:.4f}\n")
        except:
            pass

        # 6. RENDER
        transformed_points = []
        for v in vertices:
            p = mat_vec_mul(cur_rotation_matrix, v)
            transformed_points.append(p)

        for edge in edges:
            start_2d = project(transformed_points[edge[0]], cur_scale, cur_cx, cur_cy)
            end_2d = project(transformed_points[edge[1]], cur_scale, cur_cx, cur_cy)
            pygame.draw.line(screen, (255, 255, 255), start_2d, end_2d, 4)

        font = pygame.font.SysFont("Arial", 24)
        tsurf = font.render(mode_text, True, line_color)
        screen.blit(tsurf, (20, 20))
        
        instr = font.render("Fists = Clutch | Peace = Tumble | Open = Move & Roll", True, (200, 200, 200))
        screen.blit(instr, (20, 50))

        pygame.display.flip()
        clock.tick(60)

cap.release()
pygame.quit()