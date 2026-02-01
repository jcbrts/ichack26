import time
import cv2
import math
import pyautogui
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.core.base_options import BaseOptions
import numpy as np
from mediapipe.tasks.python.vision.core import image as mp_image_module

# --- 1. CONFIGURATION ---
pyautogui.PAUSE = 0
ZOOM_MULTIPLIER = 300 
PINCH_THRESHOLD = 0.005
MODEL_PATH = "/Users/jacobroberts/Desktop/ICHack/ichack26/Jacob files/hand_landmarker.task"

# SMOOTHING FACTOR
# 1 = No smoothing (jittery)
# 5-7 = Good balance
# 10+ = Very smooth but "laggy"
SMOOTHING = 5 

screen_w, screen_h = pyautogui.size()

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          
    (0, 5), (5, 6), (6, 7), (7, 8),          
    (0, 9), (9, 10), (10, 11), (11, 12),     
    (0, 13), (13, 14), (14, 15), (15, 16),   
    (0, 17), (17, 18), (18, 19), (19, 20),   
    (5, 9), (9, 13), (13, 17), (17, 5),      
]

cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit(1)

time.sleep(2)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

hand_landmarker = HandLandmarker.create_from_options(options)

previous_distance = None
failure_count = 0

# Variables for smoothing logic
curr_x, curr_y = 0, 0

while cap.isOpened():
    ret, frame = cap.read()

    if not ret or frame is None:
        failure_count += 1
        if failure_count > 10: break
        continue
    failure_count = 0

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp_image_module.Image(
        image_format=mp_image_module.ImageFormat.SRGB, data=np.array(rgb_frame)
    )

    result = hand_landmarker.detect(mp_image)

    if result.hand_landmarks:
        h, w, _ = frame.shape
        
        primary_hand = result.hand_landmarks[0]
        thumb = primary_hand[4]
        index = primary_hand[8]
        
        # --- 1. SMOOTH MOUSE MOVEMENT ---
        # Get the "target" position (where your finger actually is)
        target_x = index.x * screen_w
        target_y = index.y * screen_h
        
        # Math to blend current position with target position
        curr_x = curr_x + (target_x - curr_x) / SMOOTHING
        curr_y = curr_y + (target_y - curr_y) / SMOOTHING
        
        # Move to the smoothed position instead of the raw target
        pyautogui.moveTo(curr_x, curr_y)

        # --- 2. ZOOM LOGIC ---
        distance = math.sqrt((index.x - thumb.x)**2 + (index.y - thumb.y)**2)
        
        if previous_distance is not None:
            delta = distance - previous_distance
            if abs(delta) > PINCH_THRESHOLD:
                scroll_amount = int(delta * ZOOM_MULTIPLIER)
                pyautogui.scroll(scroll_amount)
                
        previous_distance = distance

        # --- DRAWING ---
        for hand_landmarks in result.hand_landmarks:
            points = []
            for landmark in hand_landmarks:
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                points.append((cx, cy))
            
            for start, end in HAND_CONNECTIONS:
                if start < len(points) and end < len(points):
                    cv2.line(frame, points[start], points[end], (255, 0, 0), 2)
            
            for cx, cy in points:
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            t_pt = points[4]
            i_pt = points[8]
            cv2.circle(frame, t_pt, 8, (0, 0, 255), -1) 
            cv2.circle(frame, i_pt, 8, (0, 0, 255), -1) 

    else:
        previous_distance = None

    cv2.imshow("MediaPipe Hands + Zoom", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()