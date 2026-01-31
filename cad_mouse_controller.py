import cv2
import mediapipe as mp
import time
import pyautogui

# --- SETTINGS ---
SMOOTHING = 8 
pyautogui.FAILSAFE = False

# 1. Setup MediaPipe & Drawing Utils
model_path = 'hand_landmarker.task'
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Define the lines between dots manually for the new API
# This represents the skeleton of a hand
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # Index
    (0, 9), (9, 10), (10, 11), (11, 12), # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17)            # Palm
]

latest_result = None
prev_x, prev_y = 0, 0

def result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=result_callback
)

cap = cv2.VideoCapture(0)
screen_w, screen_h = pyautogui.size()

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape # Get video dimensions
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        landmarker.detect_async(mp_image, int(time.time() * 1000))

        if latest_result and latest_result.hand_landmarks:
            for hand_landmarks in latest_result.hand_landmarks:
                
                # --- DRAWING THE SKELETON (LINES) ---
                for connection in CONNECTIONS:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    
                    # Convert normalized coordinates (0-1) to pixel coordinates
                    start_point = (int(hand_landmarks[start_idx].x * w), int(hand_landmarks[start_idx].y * h))
                    end_point = (int(hand_landmarks[end_idx].x * w), int(hand_landmarks[end_idx].y * h))
                    
                    cv2.line(frame, start_point, end_point, (0, 255, 0), 2) # Green line

                # --- DRAWING THE LANDMARKS (DOTS) ---
                for landmark in hand_landmarks:
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1) # Red dots for contrast

                # --- SHAKA PAN LOGIC ---
                thumb_out = hand_landmarks[4].x > hand_landmarks[3].x
                pinky_out = hand_landmarks[20].y < hand_landmarks[18].y
                if thumb_out and pinky_out and hand_landmarks[8].y > hand_landmarks[6].y:
                    target_x = hand_landmarks[9].x * screen_w
                    target_y = hand_landmarks[9].y * screen_h
                    curr_x = prev_x + (target_x - prev_x) / SMOOTHING
                    curr_y = prev_y + (target_y - prev_y) / SMOOTHING
                    pyautogui.dragTo(int(curr_x), int(curr_y), button='left', _pause=False)
                    prev_x, prev_y = curr_x, curr_y

        cv2.imshow('Shaka Tracker with Skeleton', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()