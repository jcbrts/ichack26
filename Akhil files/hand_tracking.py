import time
import cv2
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.core.base_options import BaseOptions

import numpy as np
from mediapipe.tasks.python.vision.core import image as mp_image_module

# Standard MediaPipe hand connections (21 landmarks)
HAND_CONNECTIONS = [
    # Palm
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),  # Thumb
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),  # Index
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),  # Middle
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),  # Ring
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),  # Pinky
    # Palm base connections
    (5, 9),
    (9, 13),
    (13, 17),
    (17, 5),
]

MODEL_PATH = "hand_landmarker.task"

# 1. Open Camera with AVFoundation (macOS fix)
cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)

# 2. Force a specific resolution (prevents black screen/timeouts on M1/M2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit(1)

# Add a short delay to allow the camera to warm up
time.sleep(2)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

hand_landmarker = HandLandmarker.create_from_options(options)

# Retry counter for black frames
failure_count = 0

while cap.isOpened():
    ret, frame = cap.read()

    # 3. Robust Error Handling: Don't quit immediately on a black frame
    if not ret or frame is None:
        failure_count += 1
        print(f"Warning: Empty frame (Attempt {failure_count})")
        if failure_count > 10:
            print("Error: Camera stopped sending data.")
            break
        continue

    # Reset failure count if we get a good frame
    failure_count = 0

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp_image_module.Image(
        image_format=mp_image_module.ImageFormat.SRGB, data=np.array(rgb_frame)
    )

    result = hand_landmarker.detect(mp_image)

    if result.hand_landmarks:
        h, w, _ = frame.shape
        for hand_landmarks in result.hand_landmarks:
            points = []
            for landmark in hand_landmarks:
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                points.append((cx, cy))
            # Draw connections first
            for start, end in HAND_CONNECTIONS:
                if start < len(points) and end < len(points):
                    cv2.line(frame, points[start], points[end], (255, 0, 0), 2)
            # Draw dots on top
            for cx, cy in points:
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    cv2.imshow("MediaPipe Hands", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
