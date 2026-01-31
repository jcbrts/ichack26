import cv2
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Setup the Task with the model you downloaded
model_path = 'hand_landmarker.task'  # Ensure this file is in your 'IC Hack 26' folder

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# This variable will store the latest results
latest_result = None

# Callback function to receive detection results asynchronously
def result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

# 2. Configure the hand landmarker options
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=result_callback
)

# 3. Open webcam and start detection
cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Flip the frame for a more natural "mirror" view
        frame = cv2.flip(frame, 1)
        
        # Convert BGR (OpenCV) to RGB (MediaPipe)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        # Detect landmarks (requires a timestamp in milliseconds)
        timestamp = int(time.time() * 1000)
        landmarker.detect_async(mp_image, timestamp)

        # Draw the results if any exist
        if latest_result and latest_result.hand_landmarks:
            for hand_landmarks in latest_result.hand_landmarks:
                # Note: Manual drawing is required because mp.solutions.drawing_utils is deprecated
                for landmark in hand_landmarks:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        cv2.imshow('MediaPipe Hand Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()