import time
import cv2
import math
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core import image as mp_image_module

# --- CONFIGURATION ---
CSV_PATH = "/Users/jacobroberts/Desktop/ICHack/ichack26/zoom_data.csv"
MODEL_PATH = "hand_landmarker.task"

# SENSITIVITY SETTINGS
THRESHOLD_MOTION = 0.005
SENS_PAN = 2.0
SENS_ZOOM = 3.0
SENS_ROT = 1.5

cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
hand_landmarker = HandLandmarker.create_from_options(options)

# State Variables
prev_dist = None
prev_angle = None
prev_mid_x = None
prev_mid_y = None

print(f"Writing Thumb+Middle data to: {CSV_PATH}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp_image_module.Image(
        image_format=mp_image_module.ImageFormat.SRGB, data=np.array(rgb_frame)
    )

    result = hand_landmarker.detect(mp_image)
    
    out_x, out_y = 0.0, 0.0
    out_zoom = 1.0
    out_rot = 0.0

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        
        # --- KEY CHANGE HERE ---
        thumb = hand[4]        # Thumb Tip
        middle = hand[12]      # Middle Finger Tip (Was hand[8] for Index)
        
        # 1. CALCULATE RAW VALUES (Using 'middle' instead of 'index')
        dist = math.sqrt((middle.x - thumb.x)**2 + (middle.y - thumb.y)**2)
        mid_x = (thumb.x + middle.x) / 2
        mid_y = (thumb.y + middle.y) / 2
        angle = math.atan2(middle.y - thumb.y, middle.x - thumb.x)

        # 2. CALCULATE DELTAS
        if prev_dist is not None:
            # Zoom
            delta_dist = dist - prev_dist
            if abs(delta_dist) > THRESHOLD_MOTION:
                out_zoom = 1.0 - (delta_dist * SENS_ZOOM)
                out_zoom = max(0.8, min(1.2, out_zoom))

            # Pan
            delta_x = mid_x - prev_mid_x
            delta_y = mid_y - prev_mid_y
            if abs(delta_x) > THRESHOLD_MOTION: out_x = delta_x * SENS_PAN
            if abs(delta_y) > THRESHOLD_MOTION: out_y = -(delta_y * SENS_PAN)

            # Rotation
            delta_angle = angle - prev_angle
            if delta_angle > math.pi: delta_angle -= 2*math.pi
            if delta_angle < -math.pi: delta_angle += 2*math.pi
            
            if abs(delta_angle) > THRESHOLD_MOTION:
                out_rot = delta_angle * SENS_ROT

        # Update State
        prev_dist = dist
        prev_mid_x, prev_mid_y = mid_x, mid_y
        prev_angle = angle
        
        # Visualization
        h, w, _ = frame.shape
        tx, ty = int(thumb.x * w), int(thumb.y * h)
        mx, my = int(middle.x * w), int(middle.y * h) # Middle Finger coords
        cx, cy = int(mid_x * w), int(mid_y * h)       # Center pivot
        
        # Draw line between Thumb and Middle Finger
        cv2.line(frame, (tx, ty), (mx, my), (255, 0, 0), 2)
        cv2.circle(frame, (tx, ty), 8, (0, 0, 255), -1) # Red Thumb
        cv2.circle(frame, (mx, my), 8, (0, 0, 255), -1) # Red Middle
        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1) # Yellow Pivot

    else:
        prev_dist = None
        prev_angle = None

    # Write to CSV
    csv_line = f"{out_x:.4f},{out_y:.4f},{out_zoom:.4f},{out_rot:.4f}"
    try:
        with open(CSV_PATH, "w") as f:
            f.write(csv_line)
    except: pass

    cv2.imshow("Thumb + Middle Control", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()