# Download the MediaPipe hand landmark model if not present
import os
import urllib.request

MODEL_URL = "https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task"
MODEL_PATH = "hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmark model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")
else:
    print("Model already exists.")
