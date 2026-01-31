import cv2
import mediapipe as mp
import time

# 1. Setup paths and models
model_path = 'hand_landmarker.task'
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# This prints to your terminal when a hand is seen
def result_callback(result, output_image, timestamp_ms):
    if result.hand_landmarks:
        print(f"Hand detected! Number of hands: {len(result.hand_landmarks)}")

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=result_callback
)

# 2. Start Camera with a "Warm-up" wait
cap = cv2.VideoCapture(0)
time.sleep(2.0) # WAIT for the camera hardware to turn on

if not cap.isOpened():
    print("ERROR: Camera not found. Try changing (0) to (1)")
    exit()

print("Camera opened. Look for the 'MediaPipe Test' window.")

# 3. Processing Loop
with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        # Convert and detect
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        landmarker.detect_async(mp_image, int(time.time() * 1000))

        # Show the video feed
        cv2.imshow('MediaPipe Test', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()