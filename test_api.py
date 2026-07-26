import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=dummy_frame)
result = detector.detect(mp_image)

print("hand_landmarks:", hasattr(result, 'hand_landmarks'))
print("handedness:", hasattr(result, 'handedness'))
if len(result.hand_landmarks) > 0:
    print(result.hand_landmarks[0][0].x)
else:
    print("No hands found (expected)")
