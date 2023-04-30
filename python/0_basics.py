# Yolo basics
# Tauno Erik
# 28.04.2023
# Source: https://www.youtube.com/watch?v=WgPbbWmnXJ8

from ultralytics import YOLO
import cv2

MODEL_FILE = '../models/klotsid_2_v4.pt'
FRAME = '../images/WIN_20230427_12_26_14_Pro.jpg'

model = YOLO(MODEL_FILE)
model.fuse()
results = model(FRAME, show=True)
cv2.waitKey(0)
