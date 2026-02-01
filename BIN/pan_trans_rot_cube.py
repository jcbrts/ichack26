import cv2
import mediapipe as mp
import math
import pygame
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from mediapipe.tasks.python.core.base_options import BaseOptions

# --- CONFIGURATION ---
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
MODEL_PATH = "/Users/jacobroberts/Desktop/ICHack/ichack26/Jacob files/hand_landmarker.task"

# SENSITIVITY SETTINGS
ROTATION_SPEED = 3.5 
SMOOTH_FACTOR = 0.15 
ZOOM_MULTIPLIER = 800  # How big the cube gets when you open your hand

# --- PYGAME SETUP ---
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("3D Hand Controller: Pan, Zoom, Rotate")
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

def project(point, scale, cx, cy):
    # Projects 3D point to 2D screen, applying Scale (Zoom) and Center (Pan)
    x, y, z = point
    # Perspective division
    f = scale / (4 + z) 
    return (int(x * f + cx), int(y * f + cy))

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
cur_yaw, cur_pitch = 0.0, 0.0
tgt_yaw, tgt_pitch = 0.0, 0.0

# Initial Transform State
cur_scale = 200.0
cur_cx, cur_cy = WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2

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

        # Render Background
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
        screen.blit(frame_surface, (0, 0))

        active_control = False
        
        if len(result.hand_landmarks) == 2:
            active_control = True
            
            # 1. Assign Hands
            anchor_hand = result.hand_landmarks[0]   # Left Hand
            rotator_hand = result.hand_landmarks[1]  # Right Hand

            # 2. PAN LOGIC (Anchor Hand Position)
            # Use Index Finger Tip (Index 8) of Anchor Hand
            ax, ay = anchor_hand[8].x, anchor_hand[8].y
            tgt_cx = int(ax * WINDOW_WIDTH)
            tgt_cy = int(ay * WINDOW_HEIGHT)

            # 3. ROTATION LOGIC (Relative Position)
            rx, ry = rotator_hand[8].x, rotator_hand[8].y
            dx = (rx - ax) 
            dy = (ry - ay)
            tgt_yaw = dx * ROTATION_SPEED * math.pi
            tgt_pitch = dy * ROTATION_SPEED * math.pi

            # 4. ZOOM LOGIC (Pinch on Rotator Hand)
            # Distance between Thumb (4) and Index (8)
            t = rotator_hand[4]
            i = rotator_hand[8]
            pinch_dist = math.sqrt((t.x - i.x)**2 + (t.y - i.y)**2)
            tgt_scale = pinch_dist * ZOOM_MULTIPLIER

            # Visualization: Draw line between hands
            pygame.draw.line(screen, (255, 255, 255), (tgt_cx, tgt_cy), 
                             (int(rx * WINDOW_WIDTH), int(ry * WINDOW_HEIGHT)), 2)

        else:
            # If hands are lost, keep cube in center but don't reset rotation
            tgt_cx, tgt_cy = WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2
            tgt_scale = 200.0
            # tgt_yaw/pitch remain as last known values to prevent snapping

        # 5. SMOOTHING (Interpolation)
        cur_cx += (tgt_cx - cur_cx) * SMOOTH_FACTOR
        cur_cy += (tgt_cy - cur_cy) * SMOOTH_FACTOR
        cur_scale += (tgt_scale - cur_scale) * SMOOTH_FACTOR
        cur_yaw += (tgt_yaw - cur_yaw) * SMOOTH_FACTOR
        cur_pitch += (tgt_pitch - cur_pitch) * SMOOTH_FACTOR

        # 6. RENDER CUBE
        transformed_points = []
        for v in vertices:
            p = rotate_y(v, cur_yaw)
            p = rotate_x(p, cur_pitch)
            transformed_points.append(p)

        cube_color = (0, 255, 255) if active_control else (100, 100, 100)
        
        for edge in edges:
            # Project using Dynamic Scale and Center
            start_2d = project(transformed_points[edge[0]], cur_scale, cur_cx, cur_cy)
            end_2d = project(transformed_points[edge[1]], cur_scale, cur_cx, cur_cy)
            pygame.draw.line(screen, cube_color, start_2d, end_2d, 4)

        # Draw Center Point (Visual feedback for Pan)
        pygame.draw.circle(screen, (255, 0, 255), (int(cur_cx), int(cur_cy)), 8)

        # UI Text
        font = pygame.font.SysFont("Arial", 20)
        if active_control:
            status = f"Scale: {int(cur_scale)} | Pitch: {int(math.degrees(cur_pitch))} | Yaw: {int(math.degrees(cur_yaw))}"
        else:
            status = "Waiting for TWO hands..."
            
        text_surf = font.render(status, True, (255, 255, 255))
        screen.blit(text_surf, (20, 20))

        pygame.display.flip()
        clock.tick(60)

cap.release()
pygame.quit()