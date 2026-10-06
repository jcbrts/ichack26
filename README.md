# AirCAD

Hands-free control of a CAD viewport with hand gestures. Built at **IC Hack 26** (Imperial College London).

A webcam tracks both hands with [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker). The distance, midpoint and angle between your index fingertips drive pan, zoom, roll and tumble of a 3D view, and the resulting camera transform is passed on to Autodesk Fusion 360.

## Gestures

| Gesture (both hands) | Action |
| --- | --- |
| Open hands, move | Pan |
| Open hands, move apart / together | Zoom |
| Open hands, rotate the line between them | Roll |
| Peace signs, move | 3D tumble |
| Two fists | Clutch (freeze, reposition without moving the view) |

## Layout

```
src/hand_controller.py     Gesture controller: webcam -> MediaPipe -> pygame cube preview,
                           writes pan/zoom/rotation to fusion_data.csv every frame
fusion/aircadfusion.py     Fusion 360 script: listens on UDP 127.0.0.1:5005 and drives the viewport camera
tools/download_hand_model.py   Downloads hand_landmarker.task into the repo root
archive/                   Earlier prototypes and experiments, kept for reference
```

## Running the controller

```bash
pip install opencv-python mediapipe pygame
python tools/download_hand_model.py     # one-off, fetches hand_landmarker.task
python src/hand_controller.py           # Esc to quit
```

The webcam index is set in `cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)` (macOS). Change `1` to `0` if you only have the built-in camera. Tuning constants (`ROTATION_SENSITIVITY`, smoothing factors, etc.) are at the top of the script.

## Fusion 360 side

`fusion/aircadfusion.py` is a Fusion script that reads `pan_x,pan_y,zoom,pitch,yaw` packets over UDP port 5005 and applies them to the active viewport. Note that the controller currently writes its output to `fusion_data.csv` rather than sending UDP, so a small bridge between the two is needed.

## Archive

`archive/` holds the prototypes that led to the final controller: mouse control via `pyautogui`, standalone zoom tracking (`*_akhil.py`), early cube/rotation tests, and `cubev3.py`, the version before the clutch and smoothed tumble were added.
