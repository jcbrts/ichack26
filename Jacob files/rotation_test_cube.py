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

# SENSITIVITY
# Lower = You have to move hands further to rotate
# Higher = Small movements rotate the cube a lot
ROTATION_SPEED = 3.0 

# SMOOTHING (Low Pass Filter)
# 0.1 = Very smooth/slow, 0.9 = Instant/jittery
SMOOTH_FACTOR = 0.2 

# --- PYGAME SETUP ---
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Direct Control 3D Cube")
clock = pygame.time.Clock()

# Cube Definition
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

def project(point):
    # Simple perspective projection
    x, y, z = point
    f = 400 / (4 + z) # Scale factor
    cx, cy = WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2
    return (int(x * f + cx), int(y * f + cy))

# --- INIT MEDIAPIPE ---
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    num_hands=2,
    running_mode=RunningMode.VIDEO,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open Camera
cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)

# State Variables
current_pitch = 0.0
current_yaw = 0.0
target_pitch = 0.0
target_yaw = 0.0

with HandLandmarker.create_from_options(options) as landmarker:
    running = True
    while running:
        # 1. Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # 2. Camera Capture
        ret, frame = cap.read()
        if not ret: break

        # Flip for "mirror" feel
        frame = cv2.flip(frame, 1)
        
        # Prepare for MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = landmarker.detect_for_video(mp_image, int(pygame.time.get_ticks()))

        # 3. Render Camera Background to Pygame
        # Convert BGR (OpenCV) to RGB (Pygame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Create a surface (transposed because Pygame expects width,height but numpy is height,width)
        frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
        screen.blit(frame_surface, (0, 0))

        # 4. Logic
        active_control = False
        
        if len(result.hand_landmarks) == 2:
            active_control = True
            
            # Identify Hands (Naive: First detected is Anchor, Second is Rotator)
            hand1 = result.hand_landmarks[0][8] # Index Tip
            hand2 = result.hand_landmarks[1][8] # Index Tip

            # Get Coordinates (0.0 to 1.0)
            x1, y1 = hand1.x, hand1.y
            x2, y2 = hand2.x, hand2.y

            # Draw "Joystick" Visualization
            h, w = WINDOW_HEIGHT, WINDOW_WIDTH
            c1 = (int(x1 * w), int(y1 * h))
            c2 = (int(x2 * w), int(y2 * h))
            
            pygame.draw.circle(screen, (0, 255, 255), c1, 15) # Cyan Anchor
            pygame.draw.circle(screen, (255, 100, 100), c2, 15) # Red Rotator
            pygame.draw.line(screen, (255, 255, 255), c1, c2, 3) # Tether

            # Calculate Deltas
            dx = (x2 - x1) 
            dy = (y2 - y1)

            # Map Distance to Angle
            target_yaw = dx * ROTATION_SPEED * math.pi
            target_pitch = dy * ROTATION_SPEED * math.pi

        # 5. Apply Smoothing (Lerp)
        # This makes the cube "float" to the target position gently
        current_yaw += (target_yaw - current_yaw) * SMOOTH_FACTOR
        current_pitch += (target_pitch - current_pitch) * SMOOTH_FACTOR

        # 6. Render Cube
        transformed_points = []
        for v in vertices:
            p = rotate_y(v, current_yaw)   # Rotate Yaw first
            p = rotate_x(p, current_pitch) # Then Pitch
            transformed_points.append(p)

        # Draw Wireframe
        cube_color = (0, 255, 0) if active_control else (100, 100, 100)
        
        for edge in edges:
            start_3d = transformed_points[edge[0]]
            end_3d = transformed_points[edge[1]]
            
            start_2d = project(start_3d)
            end_2d = project(end_3d)
            
            pygame.draw.line(screen, cube_color, start_2d, end_2d, 4)

        # Instructions
        font = pygame.font.SysFont("Arial", 24)
        if not active_control:
            msg = "Show TWO hands to grab the Joystick"
        else:
            msg = "Move Right Hand around Left Hand to Rotate"
            
        text = font.render(msg, True, (255, 255, 255))
        screen.blit(text, (20, 20))

        pygame.display.flip()
        clock.tick(60)

cap.release()
pygame.quit()